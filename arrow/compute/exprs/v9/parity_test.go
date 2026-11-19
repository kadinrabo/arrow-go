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
	exprs8 "github.com/apache/arrow-go/v18/arrow/compute/exprs" //nolint:staticcheck // SA1019: comparing against the deprecated v8 package
	"github.com/apache/arrow-go/v18/arrow/compute/exprs/v9"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	expr8 "github.com/substrait-io/substrait-go/v8/expr"
	ext8 "github.com/substrait-io/substrait-go/v8/extensions"
	types8 "github.com/substrait-io/substrait-go/v8/types"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/extensions"
	"github.com/substrait-io/substrait-go/v9/types"
)

// filter describes an expression the way iceberg-go's toSubstraitExpr
// builds it, so the same filter can be built with substrait-go v8 and v9.
type filter struct {
	fn   string
	args []filter
	ref  []int32
	lit  any
}

type (
	date      int32
	timestamp int64
	decimal   int64 // unscaled, with precision 9 and scale 2
	nullBool  struct{}
)

func call(fn string, args ...filter) filter { return filter{fn: fn, args: args} }
func ref(path ...int32) filter              { return filter{ref: path} }
func lit(v any) filter                      { return filter{lit: v} }

// in is iceberg-go's VisitIn: or(is_in(x, set), and(is_null(x), NULL)).
func in(path []int32, values ...any) filter {
	return call("or",
		call("is_in", ref(path...), lit(values)),
		call("and", call("is_null", ref(path...)), lit(nullBool{})))
}

var funcURNs = map[string]string{
	"and": boolURN, "or": boolURN, "not": boolURN,
	"equal": cmpURN, "not_equal": cmpURN, "lt": cmpURN, "lte": cmpURN, "gt": cmpURN, "gte": cmpURN,
	"is_null": cmpURN, "is_not_null": cmpURN, "is_nan": cmpURN,
	"starts_with": strURN,
	"is_in":       icebergSetURN,
}

func decimalBytes(v decimal) []byte {
	n := decimal128.FromI64(int64(v))
	buf := make([]byte, 16)
	for i := 0; i < 8; i++ {
		buf[i] = byte(n.LowBits() >> (8 * i))
		buf[8+i] = byte(uint64(n.HighBits()) >> (8 * i))
	}
	return buf
}

func structRef(path []int32) *expr.StructFieldRef {
	out := expr.NewStructFieldRef(path[0])
	for cur, i := out, 1; i < len(path); i++ {
		cur.Child = expr.NewStructFieldRef(path[i])
		cur = cur.Child.(*expr.StructFieldRef)
	}
	return out
}

func (f filter) v9(b *expr.ExprBuilder) expr.FuncArgBuilder {
	switch {
	case f.fn != "":
		args := make([]expr.FuncArgBuilder, len(f.args))
		for i, a := range f.args {
			args[i] = a.v9(b)
		}
		return b.ScalarFunc(extensions.FunctionID{URN: funcURNs[f.fn], Name: f.fn}).Args(args...)
	case f.ref != nil:
		return b.RootRef(structRef(f.ref))
	}
	return b.Wrap(literal9(f.lit))
}

func literal9(v any) (expr.Literal, error) {
	switch v := v.(type) {
	case bool:
		return expr.NewLiteral(v, false)
	case int32:
		return expr.NewPrimitiveLiteral(v, false), nil
	case int64:
		return expr.NewPrimitiveLiteral(v, false), nil
	case float64:
		return expr.NewPrimitiveLiteral(v, false), nil
	case string:
		return expr.NewPrimitiveLiteral(v, false), nil
	case date:
		return expr.NewPrimitiveLiteral(types.Date(v), false), nil
	case timestamp:
		return expr.NewPrimitiveLiteral(types.Timestamp(v), false), nil
	case decimal:
		return expr.NewLiteral(&types.Decimal{Scale: 2, Value: decimalBytes(v), Precision: 9}, false)
	case nullBool:
		return expr.NewNullLiteral(&types.BooleanType{Nullability: types.NullabilityNullable}), nil
	case []any:
		values := make(expr.ListLiteralValue, len(v))
		for i, x := range v {
			l, err := literal9(x)
			if err != nil {
				return nil, err
			}
			values[i] = l
		}
		return expr.NewLiteral(values, false)
	}
	panic("unsupported literal")
}

func structRef8(path []int32) *expr8.StructFieldRef {
	out := expr8.NewStructFieldRef(path[0])
	for cur, i := out, 1; i < len(path); i++ {
		cur.Child = expr8.NewStructFieldRef(path[i])
		cur = cur.Child.(*expr8.StructFieldRef)
	}
	return out
}

func (f filter) v8(b *expr8.ExprBuilder) expr8.FuncArgBuilder {
	switch {
	case f.fn != "":
		args := make([]expr8.FuncArgBuilder, len(f.args))
		for i, a := range f.args {
			args[i] = a.v8(b)
		}
		return b.ScalarFunc(ext8.ID{URN: funcURNs[f.fn], Name: f.fn}).Args(args...)
	case f.ref != nil:
		return b.RootRef(structRef8(f.ref))
	}
	return b.Wrap(literal8(f.lit))
}

func literal8(v any) (expr8.Literal, error) {
	switch v := v.(type) {
	case bool:
		return expr8.NewLiteral(v, false)
	case int32:
		return expr8.NewPrimitiveLiteral(v, false), nil
	case int64:
		return expr8.NewPrimitiveLiteral(v, false), nil
	case float64:
		return expr8.NewPrimitiveLiteral(v, false), nil
	case string:
		return expr8.NewPrimitiveLiteral(v, false), nil
	case date:
		return expr8.NewPrimitiveLiteral(types8.Date(v), false), nil
	case timestamp:
		return expr8.NewPrimitiveLiteral(types8.Timestamp(v), false), nil
	case decimal:
		return expr8.NewLiteral(&types8.Decimal{Scale: 2, Value: decimalBytes(v), Precision: 9}, false)
	case nullBool:
		return expr8.NewNullLiteral(&types8.BooleanType{Nullability: types8.NullabilityNullable}), nil
	case []any:
		values := make(expr8.ListLiteralValue, len(v))
		for i, x := range v {
			l, err := literal8(x)
			if err != nil {
				return nil, err
			}
			values[i] = l
		}
		return expr8.NewNestedLiteral(values, false), nil
	}
	panic("unsupported literal")
}

var (
	dictStringType = &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}
	parityStruct   = arrow.StructOf(
		arrow.Field{Name: "x", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		arrow.Field{Name: "y", Type: arrow.BinaryTypes.String, Nullable: true})
	tsMicros   = &arrow.TimestampType{Unit: arrow.Microsecond}
	decimal9_2 = &arrow.Decimal128Type{Precision: 9, Scale: 2}

	paritySchema = arrow.NewSchema([]arrow.Field{
		{Name: "i64", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
		{Name: "s", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "ds", Type: dictStringType, Nullable: true},
		{Name: "d", Type: arrow.FixedWidthTypes.Date32, Nullable: true},
		{Name: "ts", Type: tsMicros, Nullable: true},
		{Name: "dec", Type: decimal9_2, Nullable: true},
		{Name: "st", Type: parityStruct, Nullable: true},
		{Name: "b", Type: arrow.FixedWidthTypes.Boolean, Nullable: true},
	}, nil)
)

func parityRecord(t *testing.T, mem memory.Allocator) arrow.RecordBatch {
	fromJSON := func(dt arrow.DataType, js string) arrow.Array {
		arr, _, err := array.FromJSON(mem, dt, strings.NewReader(js))
		require.NoError(t, err)
		return arr
	}

	fb := array.NewFloat64Builder(mem)
	defer fb.Release()
	fb.AppendValues([]float64{1.5, math.NaN(), 0, -2, 2.5, math.NaN(), 3, 10},
		[]bool{true, true, false, true, true, true, true, true})

	ds, err := array.DictArrayFromJSON(mem, dictStringType, `[0, 1, null, 2, 3, 4, 5, 6]`,
		`["apple", "banana", "apricot", "", "app", "cherry", "Apple"]`)
	require.NoError(t, err)

	cols := []arrow.Array{
		fromJSON(arrow.PrimitiveTypes.Int64, `[1, 2, null, 4, 5, -3, 7, 0]`),
		fb.NewArray(),
		fromJSON(arrow.BinaryTypes.String, `["apple", "banana", null, "apricot", "", "app", "cherry", "Apple"]`),
		ds,
		fromJSON(arrow.FixedWidthTypes.Date32, `[18262, 18263, null, 18300, 18000, 19000, 18262, 20000]`),
		fromJSON(tsMicros, `[1577836800000000, 1577836800000001, null, 1600000000000000, 0, 1700000000000000, 1577836800000000, -1]`),
		fromJSON(decimal9_2, `["1.25", "-3.50", null, "100.00", "1.26", "0.00", "1.24", "99.99"]`),
		fromJSON(parityStruct, `[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, null, {"x": null, "y": "d"}, {"x": 5, "y": null}, null, {"x": 1, "y": "g"}, {"x": 8, "y": "h"}]`),
		fromJSON(arrow.FixedWidthTypes.Boolean, `[true, false, null, true, null, false, true, true]`),
	}
	defer func() {
		for _, c := range cols {
			c.Release()
		}
	}()
	return array.NewRecordBatch(paritySchema, cols, 8)
}

func parityBuilders() (*expr8.ExprBuilder, *expr.ExprBuilder) {
	n8, n9 := types8.NullabilityNullable, types.NullabilityNullable
	base8 := types8.StructType{Nullability: types8.NullabilityRequired, Types: []types8.Type{
		&types8.Int64Type{Nullability: n8},
		&types8.Float64Type{Nullability: n8},
		&types8.StringType{Nullability: n8},
		&types8.StringType{Nullability: n8},
		&types8.DateType{Nullability: n8},
		&types8.TimestampType{Nullability: n8},
		&types8.DecimalType{Nullability: n8, Precision: 9, Scale: 2},
		&types8.StructType{Nullability: n8, Types: []types8.Type{
			&types8.Int32Type{Nullability: n8},
			&types8.StringType{Nullability: n8},
		}},
		&types8.BooleanType{Nullability: n8},
	}}
	base9 := types.StructType{Nullability: types.NullabilityRequired, Types: []types.Type{
		&types.Int64Type{Nullability: n9},
		&types.Float64Type{Nullability: n9},
		&types.StringType{Nullability: n9},
		&types.StringType{Nullability: n9},
		&types.DateType{Nullability: n9},
		&types.TimestampType{Nullability: n9},
		&types.DecimalType{Nullability: n9, Precision: 9, Scale: 2},
		&types.StructType{Nullability: n9, Types: []types.Type{
			&types.Int32Type{Nullability: n9},
			&types.StringType{Nullability: n9},
		}},
		&types.BooleanType{Nullability: n9},
	}}
	return &expr8.ExprBuilder{Reg: expr8.NewEmptyExtensionRegistry(collection8), BaseSchema: types8.NewRecordTypeFromStruct(base8)},
		&expr.ExprBuilder{Reg: expr.NewEmptyExtensionRegistry(testCollection), BaseSchema: types.NewRecordTypeFromStruct(base9)}
}

var collection8 = func() *ext8.Collection {
	c := ext8.GetDefaultCollectionWithNoError()
	if !c.URILoaded(exprs8.SubstraitIcebergSetFuncURN) {
		if err := c.Load(exprs8.SubstraitIcebergSetFuncURN, strings.NewReader(icebergSetYAML)); err != nil {
			panic(err)
		}
	}
	return c
}()

func cref(path ...int) compute.Expression {
	return compute.NewRef(compute.FieldRefPath(compute.FieldPath(path)))
}

func cIn(t *testing.T, valueSet string, dt arrow.DataType, path ...int) compute.Expression {
	set, _, err := array.FromJSON(memory.DefaultAllocator, dt, strings.NewReader(valueSet))
	require.NoError(t, err)
	t.Cleanup(set.Release)
	isIn := compute.NewCall("is_in", []compute.Expression{cref(path...)},
		&compute.SetOptions{ValueSet: compute.NewDatumWithoutOwning(set), NullBehavior: compute.NullMatchingMatch})
	return compute.Or(isIn, compute.And(compute.IsNull(cref(path...), false),
		compute.NullLiteral(arrow.FixedWidthTypes.Boolean)))
}

func cStartsWith(pattern string, path ...int) compute.Expression {
	return compute.NewCall("starts_with", []compute.Expression{cref(path...)},
		&compute.MatchSubstringOptions{Pattern: pattern})
}

// TestParityAcrossVersions runs each filter shape iceberg-go builds through
// exprs on substrait-go v8, through this package on substrait-go v9, and as
// a compute.Expression built directly, and requires the same result from
// all three.
func TestParityAcrossVersions(t *testing.T) {
	i64, str, dict := []int32{0}, []int32{2}, []int32{3}
	cases := []struct {
		name   string
		filter filter
		cmp    func() compute.Expression
	}{
		{"true", lit(true), func() compute.Expression { return compute.NewLiteral(true) }},
		{"equal i64", call("equal", ref(0), lit(int64(4))),
			func() compute.Expression { return compute.Equal(cref(0), compute.NewLiteral(int64(4))) }},
		{"not_equal i64", call("not_equal", ref(0), lit(int64(4))),
			func() compute.Expression { return compute.NotEqual(cref(0), compute.NewLiteral(int64(4))) }},
		{"lt i64", call("lt", ref(0), lit(int64(4))),
			func() compute.Expression { return compute.Less(cref(0), compute.NewLiteral(int64(4))) }},
		{"lte i64", call("lte", ref(0), lit(int64(4))),
			func() compute.Expression { return compute.LessEqual(cref(0), compute.NewLiteral(int64(4))) }},
		{"gt i64", call("gt", ref(0), lit(int64(4))),
			func() compute.Expression { return compute.Greater(cref(0), compute.NewLiteral(int64(4))) }},
		{"gte i64", call("gte", ref(0), lit(int64(4))),
			func() compute.Expression { return compute.GreaterEqual(cref(0), compute.NewLiteral(int64(4))) }},
		{"equal string", call("equal", ref(2), lit("apple")),
			func() compute.Expression { return compute.Equal(cref(2), compute.NewLiteral("apple")) }},
		{"equal dictionary string", call("equal", ref(3), lit("apple")),
			func() compute.Expression { return compute.Equal(cref(3), compute.NewLiteral("apple")) }},
		{"gte date", call("gte", ref(4), lit(date(18263))), func() compute.Expression {
			return compute.GreaterEqual(cref(4), compute.NewLiteral(scalar.NewDate32Scalar(18263)))
		}},
		{"lt timestamp", call("lt", ref(5), lit(timestamp(1600000000000000))), func() compute.Expression {
			return compute.Less(cref(5), compute.NewLiteral(scalar.NewTimestampScalar(1600000000000000, tsMicros)))
		}},
		{"gt decimal", call("gt", ref(6), lit(decimal(125))), func() compute.Expression {
			return compute.Greater(cref(6), compute.NewLiteral(scalar.NewDecimal128Scalar(decimal128.FromI64(125), decimal9_2)))
		}},
		{"equal nested struct field", call("equal", ref(7, 0), lit(int32(1))),
			func() compute.Expression { return compute.Equal(cref(7, 0), compute.NewLiteral(int32(1))) }},
		{"is_null nested struct field", call("is_null", ref(7, 1)),
			func() compute.Expression { return compute.IsNull(cref(7, 1), false) }},
		{"is_null", call("is_null", ref(2)),
			func() compute.Expression { return compute.IsNull(cref(2), false) }},
		{"is_not_null", call("is_not_null", ref(2)),
			func() compute.Expression { return compute.IsValid(cref(2)) }},
		{"is_nan", call("is_nan", ref(1)),
			func() compute.Expression { return compute.NewCall("is_nan", []compute.Expression{cref(1)}, nil) }},
		{"not is_nan", call("not", call("is_nan", ref(1))), func() compute.Expression {
			return compute.Not(compute.NewCall("is_nan", []compute.Expression{cref(1)}, nil))
		}},
		{"starts_with", call("starts_with", ref(2), lit("ap")),
			func() compute.Expression { return cStartsWith("ap", 2) }},
		{"not starts_with", call("not", call("starts_with", ref(2), lit("ap"))),
			func() compute.Expression { return compute.Not(cStartsWith("ap", 2)) }},
		{"starts_with dictionary", call("starts_with", ref(3), lit("ap")),
			func() compute.Expression { return cStartsWith("ap", 3) }},
		{"in i64", in(i64, int64(1), int64(4), int64(7)),
			func() compute.Expression { return cIn(t, `[1, 4, 7]`, arrow.PrimitiveTypes.Int64, 0) }},
		{"not in i64", call("not", in(i64, int64(1), int64(4), int64(7))),
			func() compute.Expression { return compute.Not(cIn(t, `[1, 4, 7]`, arrow.PrimitiveTypes.Int64, 0)) }},
		{"in string", in(str, "apple", "cherry"),
			func() compute.Expression { return cIn(t, `["apple", "cherry"]`, arrow.BinaryTypes.String, 2) }},
		{"in dictionary string", in(dict, "apple", "cherry"),
			func() compute.Expression { return cIn(t, `["apple", "cherry"]`, arrow.BinaryTypes.String, 3) }},
		{"and or not with nulls", call("or",
			call("and", ref(8), call("gt", ref(0), lit(int64(1)))),
			call("not", call("lt", ref(1), lit(float64(3))))), func() compute.Expression {
			return compute.Or(
				compute.And(cref(8), compute.Greater(cref(0), compute.NewLiteral(int64(1)))),
				compute.Not(compute.Less(cref(1), compute.NewLiteral(float64(3)))))
		}},
	}

	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	rec := parityRecord(t, mem)
	defer rec.Release()
	input := compute.NewDatumWithoutOwning(rec)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b8, b9 := parityBuilders()
			e8, err := tc.filter.v8(b8).(expr8.Builder).BuildExpr()
			require.NoError(t, err)
			e9, err := tc.filter.v9(b9).(expr.Builder).BuildExpr()
			require.NoError(t, err)

			want, err := exprs8.ExecuteScalarExpression(
				exprs8.WithExtensionIDSet(ctx, exprs8.NewExtensionSetDefault(b8.Reg)), paritySchema, e8, input)
			require.NoError(t, err)
			defer want.Release()

			got9, err := exprs.ExecuteScalarExpression(ctx, paritySchema, e9, b9.Reg, input)
			require.NoError(t, err)
			defer got9.Release()

			ce := tc.cmp()
			defer ce.Release()
			gotCompute, err := compute.ExecuteScalarExpression(ctx, paritySchema, ce, input)
			require.NoError(t, err)
			defer gotCompute.Release()

			assertSameDatum(t, want, got9, "exprs/v9")
			assertSameDatum(t, want, gotCompute, "compute")
		})
	}
}

func assertSameDatum(t *testing.T, want, got compute.Datum, path string) {
	require.Equal(t, want.Kind(), got.Kind(), path)
	switch w := want.(type) {
	case *compute.ArrayDatum:
		wa, ga := w.MakeArray(), got.(*compute.ArrayDatum).MakeArray()
		defer wa.Release()
		defer ga.Release()
		assert.Truef(t, array.Equal(wa, ga), "%s: exprs %v, got %v", path, wa, ga)
	case *compute.ScalarDatum:
		gs := got.(*compute.ScalarDatum).Value
		assert.Truef(t, scalar.Equals(w.Value, gs), "%s: exprs %v, got %v", path, w.Value, gs)
	}
}

// Both exprs packages build their Arrow types for Substrait's varchar and
// similar types from one definition, so a schema from one works with the
// other.
func TestExtensionTypesShared(t *testing.T) {
	ns := types8.NamedStruct{Names: []string{"v"}, Struct: types8.StructType{
		Nullability: types8.NullabilityRequired,
		Types:       []types8.Type{&types8.VarCharType{Length: 5, Nullability: types8.NullabilityNullable}},
	}}
	schema, err := exprs8.ToArrowSchema(ns, exprs8.NewDefaultExtensionSet())
	require.NoError(t, err)

	base := types.NewRecordTypeFromStruct(types.StructType{
		Nullability: types.NullabilityRequired,
		Types:       []types.Type{&types.VarCharType{Length: 5, Nullability: types.NullabilityNullable}},
	})
	ref, err := expr.NewRootFieldRef(expr.NewStructFieldRef(0), base)
	require.NoError(t, err)

	ce, err := exprs.ToComputeExpression(schema, ref, expr.ExtensionRegistry{})
	require.NoError(t, err)
	ce.Release()
}
