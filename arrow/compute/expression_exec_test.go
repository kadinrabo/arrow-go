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

package compute_test

import (
	"context"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var execExprSchema = arrow.NewSchema([]arrow.Field{
	{Name: "a", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
	{Name: "b", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "s", Type: arrow.StructOf(
		arrow.Field{Name: "x", Type: arrow.PrimitiveTypes.Int32, Nullable: true}), Nullable: true},
}, nil)

const execExprRows = `[
	{"a": 1, "b": "foo", "s": {"x": 10}},
	{"a": 300, "b": null, "s": null},
	{"a": null, "b": "bar", "s": {"x": null}}
]`

func TestExecuteScalarExpression(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec, _, err := array.RecordFromJSON(mem, execExprSchema, strings.NewReader(execExprRows))
	require.NoError(t, err)
	defer rec.Release()

	tests := []struct {
		name     string
		expr     compute.Expression
		dt       arrow.DataType
		expected string
	}{
		{"field ref", compute.NewFieldRef("a"),
			arrow.PrimitiveTypes.Int64, `[1, 300, null]`},
		{"nested ref under a null struct", compute.NewRef(compute.FieldRefList("s", "x")),
			arrow.PrimitiveTypes.Int32, `[10, null, null]`},
		{"call", compute.Greater(compute.NewFieldRef("a"), compute.NewLiteral(int64(5))),
			arrow.FixedWidthTypes.Boolean, `[false, true, null]`},
		{"not", compute.Not(compute.IsNull(compute.NewFieldRef("b"), false)),
			arrow.FixedWidthTypes.Boolean, `[true, false, true]`},
		{"cast", compute.Cast(compute.NewRef(compute.FieldRefList("s", "x")), arrow.PrimitiveTypes.Int64),
			arrow.PrimitiveTypes.Int64, `[10, null, null]`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expected, _, err := array.FromJSON(mem, tt.dt, strings.NewReader(tt.expected))
			require.NoError(t, err)
			defer expected.Release()

			// run twice so literals and inputs survive the first run
			for i := 0; i < 2; i++ {
				out, err := compute.ExecuteScalarExpression(ctx, execExprSchema, tt.expr, compute.NewDatumWithoutOwning(rec))
				require.NoError(t, err)
				assertDatumsEqual(t, compute.NewDatumWithoutOwning(expected), out, nil, nil)
				out.Release()
			}
		})
	}
}

func TestExecuteScalarExpressionInputs(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec, _, err := array.RecordFromJSON(mem, execExprSchema, strings.NewReader(execExprRows))
	require.NoError(t, err)
	defer rec.Release()

	t.Run("missing column is null", func(t *testing.T) {
		schema := arrow.NewSchema(append(execExprSchema.Fields(),
			arrow.Field{Name: "z", Type: arrow.PrimitiveTypes.Int32, Nullable: true}), nil)
		out, err := compute.ExecuteScalarExpression(ctx, schema, compute.NewFieldRef("z"), compute.NewDatumWithoutOwning(rec))
		require.NoError(t, err)
		defer out.Release()
		assert.True(t, scalar.Equals(scalar.MakeNullScalar(arrow.PrimitiveTypes.Int32), out.(*compute.ScalarDatum).Value))
	})

	t.Run("column cast to the schema type", func(t *testing.T) {
		schema := arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Float64, Nullable: true}}, nil)
		out, err := compute.ExecuteScalarExpression(ctx, schema, compute.NewFieldRef("a"), compute.NewDatumWithoutOwning(rec))
		require.NoError(t, err)
		defer out.Release()
		assert.True(t, arrow.TypeEqual(arrow.PrimitiveTypes.Float64, out.(*compute.ArrayDatum).Type()))
	})

	t.Run("struct scalar", func(t *testing.T) {
		row, err := scalar.NewStructScalarWithNames([]scalar.Scalar{scalar.NewInt64Scalar(9)}, []string{"a"})
		require.NoError(t, err)
		expr := compute.Greater(compute.NewFieldRef("a"), compute.NewLiteral(int64(5)))
		out, err := compute.ExecuteScalarExpression(ctx, execExprSchema, expr, compute.NewDatum(row))
		require.NoError(t, err)
		defer out.Release()
		assert.True(t, scalar.Equals(scalar.NewBooleanScalar(true), out.(*compute.ScalarDatum).Value))
	})

	t.Run("struct array", func(t *testing.T) {
		arr := array.RecordToStructArray(rec)
		defer arr.Release()
		out, err := compute.ExecuteScalarExpression(ctx, execExprSchema, compute.NewFieldRef("a"), compute.NewDatumWithoutOwning(arr))
		require.NoError(t, err)
		defer out.Release()
		assert.Equal(t, int64(3), out.Len())
	})
}

func TestExecuteScalarExpressionErrors(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec, _, err := array.RecordFromJSON(mem, execExprSchema, strings.NewReader(execExprRows))
	require.NoError(t, err)
	defer rec.Release()

	tests := []struct {
		name string
		expr compute.Expression
		err  error
	}{
		{"nil expression", nil, arrow.ErrInvalid},
		{"unknown function", compute.NewCall("no_such_function", []compute.Expression{compute.NewFieldRef("a")}, nil), arrow.ErrKey},
		{"meta function other than cast", compute.NewCall("filter",
			[]compute.Expression{compute.NewFieldRef("a"), compute.NewLiteral(true)}, nil), arrow.ErrInvalid},
		{"unknown field", compute.NewFieldRef("nope"), compute.ErrNoMatch},
		{"cast that doesn't fit", compute.Cast(compute.NewFieldRef("a"), arrow.PrimitiveTypes.Int8), arrow.ErrInvalid},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := compute.ExecuteScalarExpression(ctx, execExprSchema, tt.expr, compute.NewDatumWithoutOwning(rec))
			assert.ErrorIs(t, err, tt.err)
		})
	}
}

func TestExecuteScalarExpressionFieldRefs(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	t.Run("null struct ancestors", func(t *testing.T) {
		inner := arrow.StructOf(arrow.Field{Name: "x", Type: arrow.PrimitiveTypes.Int32, Nullable: true})
		schema := arrow.NewSchema([]arrow.Field{
			{Name: "a", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
			{Name: "s", Type: arrow.StructOf(arrow.Field{Name: "t", Type: inner, Nullable: true}), Nullable: true},
		}, nil)
		// the input has no column s, so s is a null struct scalar
		rec, _, err := array.RecordFromJSON(mem, arrow.NewSchema(schema.Fields()[:1], nil), strings.NewReader(`[{"a": 1}, {"a": 2}]`))
		require.NoError(t, err)
		defer rec.Release()

		ref := compute.NewRef(compute.FieldRefList("s", "t", "x"))
		out, err := compute.ExecuteScalarExpression(ctx, schema, ref, compute.NewDatumWithoutOwning(rec))
		require.NoError(t, err)
		defer out.Release()
		assert.True(t, scalar.Equals(scalar.MakeNullScalar(arrow.PrimitiveTypes.Int32), out.(*compute.ScalarDatum).Value))

		cmp, err := compute.ExecuteScalarExpression(ctx, schema, compute.Equal(ref, compute.NewLiteral(int32(1))), compute.NewDatumWithoutOwning(rec))
		require.NoError(t, err)
		cmp.Release()
	})

	t.Run("path into a list", func(t *testing.T) {
		schema := arrow.NewSchema([]arrow.Field{{Name: "l", Type: arrow.ListOf(arrow.PrimitiveTypes.Int32), Nullable: true}}, nil)
		rec, _, err := array.RecordFromJSON(mem, schema, strings.NewReader(`[{"l": [1, 2]}, {"l": []}]`))
		require.NoError(t, err)
		defer rec.Release()

		_, err = compute.ExecuteScalarExpression(ctx, schema, compute.NewRef(compute.FieldRefPath(compute.FieldPath{0, 0})), compute.NewDatumWithoutOwning(rec))
		assert.ErrorIs(t, err, arrow.ErrInvalid)
	})

	t.Run("struct array with null rows", func(t *testing.T) {
		child, _, err := array.FromJSON(mem, arrow.PrimitiveTypes.Int64, strings.NewReader(`[1, 2, 3]`))
		require.NoError(t, err)
		defer child.Release()
		// row 1 is null while its child still holds 2
		arr, err := array.NewStructArrayWithNulls([]arrow.Array{child}, []string{"a"}, memory.NewBufferBytes([]byte{0b101}), 1, 0)
		require.NoError(t, err)
		defer arr.Release()

		schema := arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Int64, Nullable: true}}, nil)
		out, err := compute.ExecuteScalarExpression(ctx, schema, compute.NewFieldRef("a"), compute.NewDatumWithoutOwning(arr))
		require.NoError(t, err)
		defer out.Release()
		got := out.(*compute.ArrayDatum).MakeArray()
		defer got.Release()
		assert.Equal(t, "[1 (null) 3]", got.String())
	})
}
