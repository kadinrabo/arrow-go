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

package exprs_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/compute/exprs/v9"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/extensions"
	"github.com/substrait-io/substrait-go/v9/types"
)

const (
	arrowExtTypesURN = "https://github.com/apache/arrow/blob/main/format/substrait/extension_types.yaml"
	icebergSetURN    = "extension:apache.iceberg:functions_set"

	// iceberg-go's table/substrait/functions_set.yaml
	icebergSetYAML = `%YAML 1.2
---
urn: extension:apache.iceberg:functions_set
scalar_functions:
  -
    name: "is_in"
    description: Checks membership of a value in a list of values
    impls:
      - args:
        - name: needle
          value: any1
        - name: haystack
          value: list<any1>
        options:
          nan_equality:
            values: [ NAN_IS_NAN, NAN_IS_NOT_NAN ]
        return: "boolean"
`
)

var (
	arithURN = extensions.SubstraitDefaultURNPrefix + "functions_arithmetic"
	cmpURN   = extensions.SubstraitDefaultURNPrefix + "functions_comparison"
	boolURN  = extensions.SubstraitDefaultURNPrefix + "functions_boolean"
	strURN   = extensions.SubstraitDefaultURNPrefix + "functions_string"

	nullable = types.NullabilityNullable

	testCollection = func() *extensions.Collection {
		c := extensions.GetDefaultCollectionWithNoError()
		if !c.URNLoaded(icebergSetURN) {
			if err := c.Load(strings.NewReader(icebergSetYAML)); err != nil {
				panic(err)
			}
		}
		return c
	}()

	testSchema = arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
		{Name: "b", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "s", Type: arrow.StructOf(
			arrow.Field{Name: "x", Type: arrow.PrimitiveTypes.Int32, Nullable: true}), Nullable: true},
		{Name: "l", Type: arrow.ListOf(arrow.PrimitiveTypes.Int32), Nullable: true},
	}, nil)

	testBaseSchema = types.NewRecordTypeFromStruct(types.StructType{
		Nullability: types.NullabilityRequired,
		Types: []types.Type{
			&types.Int64Type{Nullability: nullable},
			&types.StringType{Nullability: nullable},
			&types.StructType{Nullability: nullable, Types: []types.Type{&types.Int32Type{Nullability: nullable}}},
			&types.ListType{Nullability: nullable, Type: &types.Int32Type{Nullability: nullable}},
		},
	})

	// testBaseSchema with the names of testSchema, which a NamedStruct
	// lists depth first
	testNamedSchema = types.NamedStruct{Names: []string{"a", "b", "s", "x", "l"}, Struct: *testBaseSchema.AsStructType()}
)

const testRows = `[
	{"a": 1, "b": "foo", "s": {"x": 10}, "l": [1, 2]},
	{"a": 300, "b": null, "s": null, "l": [3]},
	{"a": null, "b": "Fob", "s": {"x": null}, "l": null}
]`

func newBuilder() *expr.ExprBuilder {
	return &expr.ExprBuilder{Reg: expr.NewEmptyExtensionRegistry(testCollection), BaseSchema: testBaseSchema}
}

// argBuilder can be a function argument or the input of a cast.
type argBuilder interface {
	expr.Builder
	expr.FuncArgBuilder
}

func col(b *expr.ExprBuilder, i int32) argBuilder {
	return b.RootRef(expr.NewStructFieldRef(i))
}

func litArg[T int32 | int64 | string](b *expr.ExprBuilder, v T) expr.FuncArgBuilder {
	return b.Literal(expr.NewPrimitiveLiteral(v, false))
}

func overflow(pref string) *types.FunctionOption {
	return &types.FunctionOption{Name: "overflow", Preference: []string{pref}}
}

func testRecord(t *testing.T, mem memory.Allocator) arrow.RecordBatch {
	rec, _, err := array.RecordFromJSON(mem, testSchema, strings.NewReader(testRows))
	require.NoError(t, err)
	return rec
}

func TestExecuteScalarExpression(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec := testRecord(t, mem)
	defer rec.Release()

	b := newBuilder()
	tests := []struct {
		name     string
		build    func() (expr.Expression, error)
		dt       arrow.DataType
		expected string
	}{
		{"field ref", b.RootRef(expr.NewStructFieldRef(0)).BuildExpr,
			arrow.PrimitiveTypes.Int64, `[1, 300, null]`},
		{"nested ref under a null struct", b.RootRef(&expr.StructFieldRef{Field: 2, Child: expr.NewStructFieldRef(0)}).BuildExpr,
			arrow.PrimitiveTypes.Int32, `[10, null, null]`},
		{"list element", b.RootRef(&expr.StructFieldRef{Field: 3, Child: expr.NewListElemRef(0)}).BuildExpr,
			arrow.PrimitiveTypes.Int32, `[1, 3, null]`},
		{"comparison", b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(col(b, 0), litArg(b, int64(5))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, true, null]`},
		{"kleene or", b.ScalarFunc(extensions.FunctionID{URN: boolURN, Name: "or"}).Args(
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(col(b, 0), litArg(b, int64(5))),
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "is_null"}).Args(col(b, 1))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, true, null]`},
		{"three-argument and", b.ScalarFunc(extensions.FunctionID{URN: boolURN, Name: "and"}).Args(
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(col(b, 0), litArg(b, int64(0))),
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(col(b, 0), litArg(b, int64(1))),
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(col(b, 0), litArg(b, int64(2)))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, true, null]`},
		{"one-argument or", b.ScalarFunc(extensions.FunctionID{URN: boolURN, Name: "or"}).Args(
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(col(b, 0), litArg(b, int64(5)))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, true, null]`},
		{"not", b.ScalarFunc(extensions.FunctionID{URN: boolURN, Name: "not"}).Args(
			b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "is_null"}).Args(col(b, 1))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[true, false, true]`},
		{"checked add", b.ScalarFunc(extensions.FunctionID{URN: arithURN, Name: "add"}, overflow("ERROR")).Args(col(b, 0), litArg(b, int64(1))).BuildExpr,
			arrow.PrimitiveTypes.Int64, `[2, 301, null]`},
		{"is_in", b.ScalarFunc(extensions.FunctionID{URN: icebergSetURN, Name: "is_in"}).Args(col(b, 1),
			b.Wrap(expr.NewLiteral(expr.ListLiteralValue{expr.NewPrimitiveLiteral("foo", false), expr.NewPrimitiveLiteral("bar", false)}, false))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[true, false, false]`},
		{"starts_with", b.ScalarFunc(extensions.FunctionID{URN: strURN, Name: "starts_with"}).Args(col(b, 1), litArg(b, "fo")).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[true, null, false]`},
		{"starts_with ignoring case", b.ScalarFunc(extensions.FunctionID{URN: strURN, Name: "starts_with"},
			&types.FunctionOption{Name: "case_sensitivity", Preference: []string{"CASE_INSENSITIVE"}}).Args(col(b, 1), litArg(b, "fo")).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[true, null, true]`},
		{"cast", b.Cast(b.RootRef(&expr.StructFieldRef{Field: 2, Child: expr.NewStructFieldRef(0)}), &types.Int64Type{Nullability: nullable}).
			FailBehavior(types.CastFailBehaviorThrowException).BuildExpr,
			arrow.PrimitiveTypes.Int64, `[10, null, null]`},
		{"is_in an empty list", b.ScalarFunc(extensions.FunctionID{URN: icebergSetURN, Name: "is_in"}).Args(col(b, 1),
			b.Literal(expr.NewEmptyListLiteral(&types.StringType{Nullability: nullable}, false))).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, false, false]`},
		{"whole struct ref", b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "is_null"}).Args(col(b, 2)).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, true, false]`},
		{"whole list ref", b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "is_null"}).Args(col(b, 3)).BuildExpr,
			arrow.FixedWidthTypes.Boolean, `[false, false, true]`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := tt.build()
			require.NoError(t, err)

			expected, _, err := array.FromJSON(mem, tt.dt, strings.NewReader(tt.expected))
			require.NoError(t, err)
			defer expected.Release()

			out, err := exprs.ExecuteScalarExpression(ctx, testSchema, e, b.Reg, compute.NewDatumWithoutOwning(rec))
			require.NoError(t, err)
			defer out.Release()

			got := out.(*compute.ArrayDatum).MakeArray()
			defer got.Release()
			assert.Truef(t, array.Equal(expected, got), "expected: %s\ngot: %s", expected, got)
		})
	}
}

func TestExecuteScalarSubstrait(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	// the input lacks a and l, and has its columns in another order
	schema := arrow.NewSchema([]arrow.Field{testSchema.Field(2), testSchema.Field(1)}, nil)
	rec, _, err := array.RecordFromJSON(mem, schema, strings.NewReader(
		`[{"s": {"x": 10}, "b": "foo"}, {"s": null, "b": null}, {"s": {"x": 3}, "b": "bar"}]`))
	require.NoError(t, err)
	defer rec.Release()

	b := newBuilder()
	gt, err := b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "gt"}).Args(
		b.RootRef(&expr.StructFieldRef{Field: 2, Child: expr.NewStructFieldRef(0)}), litArg(b, int32(5))).BuildExpr()
	require.NoError(t, err)
	e := &expr.Extended{
		Extensions:   b.Reg.Set,
		ReferredExpr: []expr.ExpressionReference{expr.NewExpressionReference([]string{"out"}, gt)},
		BaseSchema:   testNamedSchema,
	}

	out, err := exprs.ExecuteScalarSubstrait(ctx, e, compute.NewDatumWithoutOwning(rec))
	require.NoError(t, err)
	defer out.Release()
	got := out.(*compute.ArrayDatum).MakeArray()
	defer got.Release()
	assert.Equal(t, "[true (null) false]", got.String())
}

func TestExecuteScalarSubstraitErrors(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec := testRecord(t, mem)
	defer rec.Release()

	ref, err := col(newBuilder(), 0).BuildExpr()
	require.NoError(t, err)
	er := expr.NewExpressionReference([]string{"out"}, ref)

	tests := []struct {
		name string
		e    *expr.Extended
		err  error
	}{
		{"nil", nil, arrow.ErrInvalid},
		{"no expression", &expr.Extended{BaseSchema: testNamedSchema}, arrow.ErrInvalid},
		{"two expressions", &expr.Extended{BaseSchema: testNamedSchema, ReferredExpr: []expr.ExpressionReference{er, er}},
			arrow.ErrNotImplemented},
		{"measure", &expr.Extended{BaseSchema: testNamedSchema, ReferredExpr: []expr.ExpressionReference{
			expr.NewMeasureReference([]string{"out"}, &expr.AggregateFunction{})}}, arrow.ErrNotImplemented},
		{"top-level names only", &expr.Extended{ReferredExpr: []expr.ExpressionReference{er},
			BaseSchema: types.NamedStruct{Names: []string{"a", "b", "s", "l"}, Struct: testNamedSchema.Struct}}, arrow.ErrInvalid},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := exprs.ExecuteScalarSubstrait(ctx, tt.e, compute.NewDatumWithoutOwning(rec))
			assert.ErrorIs(t, err, tt.err)
		})
	}
}

func TestToComputeExpressionReuse(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec := testRecord(t, mem)
	defer rec.Release()

	b := newBuilder()
	e, err := b.ScalarFunc(extensions.FunctionID{URN: icebergSetURN, Name: "is_in"}).Args(col(b, 0),
		b.Wrap(expr.NewLiteral(expr.ListLiteralValue{expr.NewPrimitiveLiteral(int64(1), false)}, false))).BuildExpr()
	require.NoError(t, err)

	ce, err := exprs.ToComputeExpression(testSchema, e, b.Reg)
	require.NoError(t, err)
	defer ce.Release()

	// a converted expression can run any number of times
	for i := 0; i < 2; i++ {
		out, err := compute.ExecuteScalarExpression(ctx, testSchema, ce, compute.NewDatumWithoutOwning(rec))
		require.NoError(t, err)
		got := out.(*compute.ArrayDatum).MakeArray()
		assert.Equal(t, `[true false false]`, got.String())
		got.Release()
		out.Release()
	}
}

// Substrait's and/or take any number of arguments, while Arrow's kleene
// functions take two.
func TestVariadicBooleanString(t *testing.T) {
	b := newBuilder()
	e, err := b.ScalarFunc(extensions.FunctionID{URN: boolURN, Name: "or"}).Args(
		b.ScalarFunc(extensions.FunctionID{URN: cmpURN, Name: "is_null"}).Args(col(b, 1))).BuildExpr()
	require.NoError(t, err)

	ce, err := exprs.ToComputeExpression(testSchema, e, b.Reg)
	require.NoError(t, err)
	defer ce.Release()
	assert.True(t, strings.HasPrefix(ce.String(), "is_null("), ce.String())
}

func TestArithmeticOverflow(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec := testRecord(t, mem)
	defer rec.Release()

	b := newBuilder()
	for _, tt := range []struct {
		pref string
		fn   string
		err  error
	}{
		{"SILENT", "add_unchecked(", nil},
		{"ERROR", "add(", arrow.ErrInvalid},
	} {
		t.Run(tt.pref, func(t *testing.T) {
			e, err := b.ScalarFunc(extensions.FunctionID{URN: arithURN, Name: "add"}, overflow(tt.pref)).
				Args(col(b, 0), litArg(b, int64(math.MaxInt64))).BuildExpr()
			require.NoError(t, err)

			ce, err := exprs.ToComputeExpression(testSchema, e, b.Reg)
			require.NoError(t, err)
			defer ce.Release()
			assert.True(t, strings.HasPrefix(ce.String(), tt.fn), ce.String())

			out, err := compute.ExecuteScalarExpression(ctx, testSchema, ce, compute.NewDatumWithoutOwning(rec))
			if tt.err != nil {
				assert.ErrorIs(t, err, tt.err)
				return
			}
			require.NoError(t, err)
			out.Release()
		})
	}
}

// THROW_EXCEPTION casts fail when a value doesn't fit, where exprs on
// substrait-go v8 casts unsafely and returns 44 for 300 cast to i8.
func TestCastThrowExceptionFails(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec := testRecord(t, mem)
	defer rec.Release()

	b := newBuilder()
	e, err := b.Cast(col(b, 0), &types.Int8Type{Nullability: nullable}).
		FailBehavior(types.CastFailBehaviorThrowException).BuildExpr()
	require.NoError(t, err)

	_, err = exprs.ExecuteScalarExpression(ctx, testSchema, e, b.Reg, compute.NewDatumWithoutOwning(rec))
	assert.ErrorIs(t, err, arrow.ErrInvalid)
}

func TestToComputeExpressionErrors(t *testing.T) {
	b := newBuilder()
	mustBuild := func(e expr.Expression, err error) expr.Expression {
		require.NoError(t, err)
		return e
	}
	listElemThenField := &expr.FieldReference{Root: expr.RootReference, Reference: &expr.StructFieldRef{
		Field: 3, Child: &expr.ListElementRef{Offset: 0, Child: expr.NewStructFieldRef(0)}}}
	outOfRange := &expr.FieldReference{Root: expr.RootReference, Reference: expr.NewStructFieldRef(10)}
	mapKey := &expr.FieldReference{Root: expr.RootReference, Reference: &expr.StructFieldRef{
		Field: 0, Child: expr.NewMapKeyRef(expr.NewPrimitiveLiteral("k", false))}}

	tests := []struct {
		name   string
		schema *arrow.Schema
		e      expr.Expression
		err    error
		msg    string
	}{
		{"nil expression", testSchema, nil, arrow.ErrInvalid, ""},
		{"unmapped function", testSchema, mustBuild(b.ScalarFunc(extensions.FunctionID{URN: arithURN, Name: "modulus"}).
			Args(col(b, 0), col(b, 0)).BuildExpr()), arrow.ErrNotImplemented, "modulus"},
		{"unspecified cast behavior", testSchema, mustBuild(b.Cast(col(b, 0), &types.Int8Type{Nullability: nullable}).BuildExpr()),
			arrow.ErrInvalid, "unspecified"},
		{"return null cast behavior", testSchema, mustBuild(b.Cast(col(b, 0), &types.Int8Type{Nullability: nullable}).
			FailBehavior(types.CastFailBehaviorReturnNull).BuildExpr()), arrow.ErrNotImplemented, ""},
		{"referenced field of another type", arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Int32}}, nil),
			mustBuild(b.RootRef(expr.NewStructFieldRef(0)).BuildExpr()),
			arrow.ErrInvalid, "referenced field .field(0) was int32, but should have been int64"},
		{"field reference without a schema", nil, mustBuild(b.RootRef(expr.NewStructFieldRef(0)).BuildExpr()),
			arrow.ErrInvalid, "schema"},
		{"field out of range", testSchema, outOfRange, compute.ErrIndexRange, ""},
		{"struct field of a list element", testSchema, listElemThenField, arrow.ErrNotImplemented, "list element"},
		{"map key", testSchema, mapKey, arrow.ErrNotImplemented, "MapKeyRef"},
		{"user-defined type without extensions", testSchema,
			expr.NewNullLiteral(&types.UserDefinedType{TypeReference: 1, Nullability: nullable}), arrow.ErrNotImplemented, "anchor 1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ce, err := exprs.ToComputeExpression(tt.schema, tt.e, b.Reg)
			if ce != nil {
				ce.Release()
			}
			assert.ErrorIs(t, err, tt.err)
			if tt.msg != "" {
				assert.ErrorContains(t, err, tt.msg)
			}
		})
	}
}

// Substrait numbers type anchors and type variation anchors separately, so
// one plan can give anchor 1 to both a type and a variation.
func TestTypeAndVariationAnchorsDecodeSeparately(t *testing.T) {
	set := extensions.NewSet()
	typeAnchor := set.GetTypeAnchor(extensions.TypeID{URN: arrowExtTypesURN, Name: "fp16"})
	variationAnchor := set.GetTypeVariationAnchor(extensions.TypeVariationID{URN: arrowExtTypesURN, Name: "u32"})
	require.Equal(t, typeAnchor, variationAnchor)
	reg := expr.NewExtensionRegistry(set, testCollection)

	for _, tt := range []struct {
		name string
		typ  types.Type
		dt   arrow.DataType
	}{
		{"type", &types.UserDefinedType{TypeReference: typeAnchor, Nullability: nullable}, arrow.FixedWidthTypes.Float16},
		{"variation", &types.Int32Type{TypeVariationRef: variationAnchor, Nullability: nullable}, arrow.PrimitiveTypes.Uint32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ce, err := exprs.ToComputeExpression(testSchema, expr.NewNullLiteral(tt.typ), reg)
			require.NoError(t, err)
			defer ce.Release()
			assert.Truef(t, arrow.TypeEqual(tt.dt, ce.Type()), "expected %s, got %s", tt.dt, ce.Type())
		})
	}
}
