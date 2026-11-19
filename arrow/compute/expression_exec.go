// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compute

import (
	"context"
	"fmt"
	"slices"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/apache/arrow-go/v18/arrow/internal"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// ExecuteScalarExpression evaluates a scalar expression against
// partialInput, a record batch or a struct array or scalar. Columns are
// matched to inputSchema by name: a missing column becomes a null scalar and
// a column of another type is cast to the schema's type. Field references
// are resolved against inputSchema.
//
// The expression can contain literals, field references, and calls to
// scalar functions or to "cast".
func ExecuteScalarExpression(ctx context.Context, inputSchema *arrow.Schema, expr Expression, partialInput Datum) (Datum, error) {
	if expr == nil {
		return nil, arrow.ErrInvalid
	}

	batch, err := makeExprExecBatch(ctx, inputSchema, partialInput)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, v := range batch.Values {
			v.Release()
		}
	}()

	return executeScalarExpr(ctx, inputSchema, batch, expr)
}

func executeScalarExpr(ctx context.Context, schema *arrow.Schema, input ExecBatch, expr Expression) (Datum, error) {
	switch e := expr.(type) {
	case *Literal:
		return NewDatum(e.Literal), nil
	case *Parameter:
		return execParameter(ctx, schema, input, e)
	case *Call:
		fn, ok := GetExecCtx(ctx).Registry.GetFunction(e.funcName)
		if !ok {
			return nil, fmt.Errorf("%w: function '%s' not found", arrow.ErrKey, e.funcName)
		}
		// cast is a meta function over scalar kernels, while the other meta
		// functions (filter, take, sort) change the length of their input
		if fn.Kind() != FuncScalar && fn.Name() != "cast" {
			return nil, fmt.Errorf("%w: %s is not a scalar function", arrow.ErrInvalid, e.funcName)
		}

		args := make([]Datum, len(e.args))
		for i, a := range e.args {
			arg, err := executeScalarExpr(ctx, schema, input, a)
			if err != nil {
				return nil, err
			}
			defer arg.Release()
			args[i] = arg
		}
		return fn.Execute(ctx, e.options, args...)
	}
	return nil, fmt.Errorf("%w: executing %T", arrow.ErrNotImplemented, expr)
}

func execParameter(ctx context.Context, schema *arrow.Schema, input ExecBatch, p *Parameter) (Datum, error) {
	path, err := p.ref.FindOne(schema)
	if err != nil {
		return nil, err
	}
	fieldType, err := structPathType(schema, path)
	if err != nil {
		return nil, err
	}

	switch v := input.Values[path[0]].(type) {
	case *ArrayDatum:
		cur := v.Value
		cur.Retain()
		for _, i := range path[1:] {
			next, err := exprStructField(GetAllocator(ctx), cur, i)
			cur.Release()
			if err != nil {
				return nil, err
			}
			cur = next
		}
		return &ArrayDatum{Value: cur}, nil
	case *ScalarDatum:
		sc := v.Value
		for _, i := range path[1:] {
			st, ok := sc.(*scalar.Struct)
			if !ok {
				return nil, fmt.Errorf("%w: struct field ref for non struct type %s",
					arrow.ErrInvalid, sc.DataType())
			}
			if !st.IsValid() {
				// every field under a null struct is null
				return &ScalarDatum{Value: scalar.MakeNullScalar(fieldType)}, nil
			}
			sc = st.Value[i]
		}
		return NewDatum(sc), nil
	}
	return nil, fmt.Errorf("%w: field ref into %s", arrow.ErrNotImplemented, input.Values[path[0]])
}

// structPathType returns the type of the field at path in schema. FindOne
// also resolves paths into list elements, which a field reference can't
// select, so every step must be into a struct.
func structPathType(schema *arrow.Schema, path FieldPath) (arrow.DataType, error) {
	dt := schema.Field(path[0]).Type
	for _, i := range path[1:] {
		st, ok := dt.(*arrow.StructType)
		if !ok {
			return nil, fmt.Errorf("%w: struct field ref for non struct type %s", arrow.ErrInvalid, dt)
		}
		dt = st.Field(i).Type
	}
	return dt, nil
}

// exprStructField returns field i of a struct array as a new array, sliced
// to the struct's offset and length and null wherever the struct is null.
func exprStructField(mem memory.Allocator, parent arrow.ArrayData, i int) (arrow.ArrayData, error) {
	off, n := parent.Offset(), parent.Len()
	child := array.NewSliceData(parent.Children()[i], int64(off), int64(off+n))

	if parent.NullN() == 0 || len(parent.Buffers()) == 0 || parent.Buffers()[0] == nil {
		return child, nil
	}
	parentBits := parent.Buffers()[0].Bytes()
	parentNulls := parent.NullN()
	if parentNulls == array.UnknownNullCount {
		parentNulls = n - bitutil.CountSetBits(parentBits, off, n)
	}
	if parentNulls == 0 {
		return child, nil
	}

	storage := child.DataType()
	if ext, ok := storage.(arrow.ExtensionType); ok {
		storage = ext.StorageType()
	}
	if storage.ID() == arrow.NULL {
		return child, nil
	}
	if !internal.DefaultHasValidityBitmap(storage.ID()) {
		err := fmt.Errorf("%w: cannot propagate struct nulls to field type %s", arrow.ErrNotImplemented, child.DataType())
		child.Release()
		return nil, err
	}
	defer child.Release()

	// keep the child's offset so its other buffers stay valid
	cOff := child.Offset()
	bits := memory.NewResizableBuffer(mem)
	defer bits.Release()
	bits.Resize(int(bitutil.BytesForBits(int64(cOff + n))))
	if cb := child.Buffers()[0]; cb != nil {
		bitutil.BitmapAnd(parentBits, cb.Bytes(), int64(off), int64(cOff), bits.Bytes(), int64(cOff), int64(n))
	} else {
		bitutil.CopyBitmap(parentBits, off, n, bits.Bytes(), cOff)
	}
	nulls := n - bitutil.CountSetBits(bits.Bytes(), cOff, n)

	buffers := slices.Clone(child.Buffers())
	buffers[0] = bits
	out := array.NewData(child.DataType(), n, buffers, child.Children(), nulls, cOff)
	if dict := child.Dictionary(); dict != nil {
		out.SetDictionary(dict)
	}
	return out, nil
}

func makeExprExecBatch(ctx context.Context, schema *arrow.Schema, partial Datum) (out ExecBatch, err error) {
	// cleanup if we get an error
	defer func() {
		if err != nil {
			for _, v := range out.Values {
				if v != nil {
					v.Release()
				}
			}
			out = ExecBatch{}
		}
	}()

	if partial.Kind() == KindRecord {
		partialBatch := partial.(*RecordDatum).Value
		batchSchema := partialBatch.Schema()

		out.Values = make([]Datum, schema.NumFields())
		out.Len = partialBatch.NumRows()

		for i, field := range schema.Fields() {
			idxes := batchSchema.FieldIndices(field.Name)
			switch len(idxes) {
			case 0:
				out.Values[i] = &ScalarDatum{Value: scalar.MakeNullScalar(field.Type)}
			case 1:
				col := partialBatch.Column(idxes[0])
				if !arrow.TypeEqual(col.DataType(), field.Type) {
					// referenced field was present but didn't have expected type
					// we'll cast this case for now
					col, err = CastArray(ctx, col, SafeCastOptions(field.Type))
					if err != nil {
						return out, err
					}
					defer col.Release()
				}
				out.Values[i] = NewDatum(col)
			default:
				err = fmt.Errorf("%w: exec batch field '%s' ambiguous, more than one match",
					arrow.ErrInvalid, field.Name)
				return out, err
			}
		}
		return
	}

	part, ok := partial.(ArrayLikeDatum)
	if !ok {
		return out, fmt.Errorf("%w: executing an expression against %s", arrow.ErrNotImplemented, partial)
	}

	if part.Type().ID() == arrow.STRUCT {
		switch part := part.(type) {
		case *ArrayDatum:
			// the columns carry the struct's nulls, so a null row reads as
			// null whatever its children hold
			st := part.Type().(*arrow.StructType)
			cols := make([]arrow.Array, 0, st.NumFields())
			defer func() {
				for _, c := range cols {
					c.Release()
				}
			}()
			for i := range st.NumFields() {
				child, err := exprStructField(GetAllocator(ctx), part.Value, i)
				if err != nil {
					return out, err
				}
				cols = append(cols, array.MakeFromData(child))
				child.Release()
			}

			batch := array.NewRecordBatch(arrow.NewSchema(st.Fields(), nil), cols, part.Len())
			defer batch.Release()
			return makeExprExecBatch(ctx, schema, NewDatumWithoutOwning(batch))
		case *ScalarDatum:
			out.Len = 1
			out.Values = make([]Datum, schema.NumFields())

			s := part.Value.(*scalar.Struct)
			dt := s.Type.(*arrow.StructType)

			for i, field := range schema.Fields() {
				idx, found := dt.FieldIdx(field.Name)
				if !found {
					out.Values[i] = &ScalarDatum{Value: scalar.MakeNullScalar(field.Type)}
					continue
				}

				val := s.Value[idx]
				if !arrow.TypeEqual(val.DataType(), field.Type) {
					// referenced field was present but didn't have the expected
					// type. for now we'll cast this
					val, err = val.CastTo(field.Type)
					if err != nil {
						return out, err
					}
					if !s.Value[idx].IsValid() {
						// CastTo of a null returns a new scalar, which the datum owns
						out.Values[i] = &ScalarDatum{Value: val}
						continue
					}
				}
				out.Values[i] = NewDatum(val)
			}
			return
		}
	}

	return out, fmt.Errorf("%w: executing an expression against %s", arrow.ErrNotImplemented, partial)
}
