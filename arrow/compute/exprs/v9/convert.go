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

//go:build go1.18

package exprs

import (
	"context"
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/types"
)

// ExecuteScalarExpression converts e with ToComputeExpression and runs it
// with compute.ExecuteScalarExpression against partialInput, a record batch
// or a struct array or scalar whose columns are matched to schema by name.
func ExecuteScalarExpression(ctx context.Context, schema *arrow.Schema, e expr.Expression, ext expr.ExtensionRegistry, partialInput compute.Datum) (compute.Datum, error) {
	ce, err := ToComputeExpression(schema, e, ext)
	if err != nil {
		return nil, err
	}
	defer ce.Release()
	return compute.ExecuteScalarExpression(ctx, schema, ce, partialInput)
}

// ToComputeExpression converts the Substrait scalar expression e to a
// compute.Expression whose field references are positions in schema, as
// Substrait's are. ext resolves the type and type variation anchors e uses,
// and can be the zero value when e uses none.
//
// The result holds the converted literals. Call Release when done with it.
func ToComputeExpression(schema *arrow.Schema, e expr.Expression, ext expr.ExtensionRegistry) (compute.Expression, error) {
	if e == nil {
		return nil, arrow.ErrInvalid
	}
	if !e.IsScalar() {
		return nil, fmt.Errorf("%w: cannot convert non-scalar expression %s", arrow.ErrInvalid, e)
	}
	c := converter{schema: schema, types: typeResolver{reg: ext}}
	return c.convert(e)
}

type converter struct {
	schema *arrow.Schema
	types  typeResolver
}

func (c converter) convert(e expr.Expression) (compute.Expression, error) {
	switch e := e.(type) {
	case expr.Literal:
		d, err := literalToDatum(memory.DefaultAllocator, e, c.types)
		if err != nil {
			return nil, err
		}
		return &compute.Literal{Literal: d}, nil
	case *expr.FieldReference:
		return c.fieldRef(e)
	case *expr.Cast:
		return c.cast(e)
	case *expr.ScalarFunction:
		return c.scalarFunction(e)
	}
	return nil, fmt.Errorf("%w: converting %T", arrow.ErrNotImplemented, e)
}

func (c converter) cast(e *expr.Cast) (compute.Expression, error) {
	if e.Input == nil {
		return nil, fmt.Errorf("%w: cast without argument to cast", arrow.ErrInvalid)
	}

	dt, _, err := fromSubstraitType(e.Type, c.types)
	if err != nil {
		return nil, fmt.Errorf("%w: could not determine type for cast", err)
	}

	var opts *compute.CastOptions
	switch e.FailureBehavior {
	case types.CastFailBehaviorThrowException:
		opts = compute.SafeCastOptions(dt)
	case types.CastFailBehaviorUnspecified:
		return nil, fmt.Errorf("%w: cast behavior unspecified", arrow.ErrInvalid)
	case types.CastFailBehaviorReturnNull:
		return nil, fmt.Errorf("%w: cast behavior return nil", arrow.ErrNotImplemented)
	default:
		return nil, fmt.Errorf("%w: cast behavior %v", arrow.ErrNotImplemented, e.FailureBehavior)
	}

	in, err := c.convert(e.Input)
	if err != nil {
		return nil, err
	}
	return compute.NewCall("cast", []compute.Expression{in}, opts), nil
}

func (c converter) scalarFunction(sf *expr.ScalarFunction) (out compute.Expression, err error) {
	conv, ok := lookupScalarFunction(sf)
	if !ok {
		return nil, fmt.Errorf("%w: %s", arrow.ErrNotImplemented, sf.Name())
	}

	args := make([]compute.Expression, 0, sf.NArgs())
	defer func() {
		if err != nil {
			for _, a := range args {
				a.Release()
			}
		}
	}()
	for i := 0; i < sf.NArgs(); i++ {
		switch v := sf.Arg(i).(type) {
		case types.Enum:
			args = append(args, compute.NewLiteral(scalar.NewStringScalar(string(v))))
		case expr.Expression:
			arg, err := c.convert(v)
			if err != nil {
				return nil, err
			}
			args = append(args, arg)
		default:
			return nil, fmt.Errorf("%w: %s argument of type %T", arrow.ErrNotImplemented, sf.Name(), v)
		}
	}

	fname, callArgs, opts, err := conv(sf, args)
	if err != nil {
		return nil, err
	}
	return compute.NewCall(fname, callArgs, opts), nil
}

// fieldRef converts a direct reference from the root of the input. Struct
// fields become a field path into schema, and list elements become calls to
// list_element.
func (c converter) fieldRef(e *expr.FieldReference) (_ compute.Expression, err error) {
	if e.Root != expr.RootReference {
		return nil, fmt.Errorf("%w: only RootReference is implemented", arrow.ErrNotImplemented)
	}
	ref, ok := e.Reference.(expr.ReferenceSegment)
	if !ok {
		return nil, fmt.Errorf("%w: only direct references are implemented", arrow.ErrNotImplemented)
	}
	if c.schema == nil {
		return nil, fmt.Errorf("%w: field reference without an input schema", arrow.ErrInvalid)
	}

	sref, ok := ref.(*expr.StructFieldRef)
	if !ok || sref.Field < 0 || int(sref.Field) >= c.schema.NumFields() {
		return nil, arrow.ErrInvalid
	}
	path := compute.FieldPath{int(sref.Field)}
	dt := c.schema.Field(int(sref.Field)).Type

	// elem holds the list_element calls once the reference steps into a list
	var elem compute.Expression
	defer func() {
		if err != nil && elem != nil {
			elem.Release()
		}
	}()
	for ref = ref.GetChild(); ref != nil; ref = ref.GetChild() {
		switch r := ref.(type) {
		case *expr.StructFieldRef:
			if elem != nil {
				return nil, fmt.Errorf("%w: struct field of a list element", arrow.ErrNotImplemented)
			}
			st, ok := dt.(*arrow.StructType)
			if !ok {
				return nil, fmt.Errorf("%w: struct field ref for non struct type %s", arrow.ErrInvalid, dt)
			}
			if r.Field < 0 || int(r.Field) >= st.NumFields() {
				return nil, fmt.Errorf("%w: indices=%s", compute.ErrIndexRange, ref)
			}
			path = append(path, int(r.Field))
			dt = st.Field(int(r.Field)).Type
		case *expr.ListElementRef:
			switch dt.ID() {
			case arrow.LIST, arrow.LARGE_LIST, arrow.FIXED_SIZE_LIST:
			default:
				return nil, fmt.Errorf("%w: cannot reference list element in non-list type %s", arrow.ErrInvalid, dt)
			}
			if elem == nil {
				elem = compute.NewRef(compute.FieldRefPath(path))
			}
			elem = compute.NewCall("list_element", []compute.Expression{
				elem, compute.NewLiteral(scalar.NewInt32Scalar(r.Offset))}, nil)
			dt = dt.(arrow.ListLikeType).Elem()
		default:
			return nil, fmt.Errorf("%w: reference segment %T", arrow.ErrNotImplemented, ref)
		}
	}

	if err := c.checkRefType(e, dt); err != nil {
		return nil, err
	}
	if elem != nil {
		return elem, nil
	}
	return compute.NewRef(compute.FieldRefPath(path)), nil
}

// checkRefType errors if the referenced field of type dt doesn't have the
// type that e declares. Substrait has no large string, large binary or
// dictionary types, so those match their small, decoded counterparts.
func (c converter) checkRefType(e *expr.FieldReference, dt arrow.DataType) error {
	if e.GetType() == nil {
		return nil
	}
	expected, _, err := fromSubstraitType(e.GetType(), c.types)
	if err != nil {
		return err
	}

	valueType := dt
	if dict, ok := dt.(*arrow.DictionaryType); ok && arrow.IsBaseBinary(dict.ValueType.ID()) {
		valueType = dict.ValueType
	}
	switch {
	case arrow.TypeEqual(valueType, expected):
	case expected.ID() == arrow.STRING && valueType.ID() == arrow.LARGE_STRING:
	case expected.ID() == arrow.BINARY && valueType.ID() == arrow.LARGE_BINARY:
	default:
		return fmt.Errorf("%w: referenced field %s was %s, but should have been %s",
			arrow.ErrInvalid, e.Reference, dt, expected)
	}
	return nil
}
