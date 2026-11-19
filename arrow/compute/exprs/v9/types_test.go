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
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute/exprs/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/extensions"
	"github.com/substrait-io/substrait-go/v9/types"
)

func TestToArrowSchema(t *testing.T) {
	set := extensions.NewSet()
	fp16 := set.GetTypeAnchor(extensions.TypeID{URN: arrowExtTypesURN, Name: "fp16"})
	u8 := set.GetTypeVariationAnchor(extensions.TypeVariationID{URN: arrowExtTypesURN, Name: "u8"})
	reg := expr.NewExtensionRegistry(set, testCollection)

	required := types.NullabilityRequired
	// the names cover the fields of structs nested in a struct, a list and
	// a map value, depth first
	base := types.NamedStruct{
		Names: []string{"a", "s", "x", "t", "y", "l", "z", "m", "v", "u", "h"},
		Struct: types.StructType{Nullability: required, Types: []types.Type{
			&types.Int64Type{Nullability: required},
			&types.StructType{Nullability: nullable, Types: []types.Type{
				&types.Int32Type{Nullability: nullable},
				&types.StructType{Nullability: nullable, Types: []types.Type{&types.StringType{Nullability: nullable}}},
			}},
			&types.ListType{Nullability: nullable, Type: &types.StructType{Nullability: nullable,
				Types: []types.Type{&types.Int32Type{Nullability: nullable}}}},
			&types.MapType{Nullability: nullable, Key: &types.StringType{Nullability: required},
				Value: &types.StructType{Nullability: nullable, Types: []types.Type{&types.Float64Type{Nullability: nullable}}}},
			&types.Int8Type{Nullability: nullable, TypeVariationRef: u8},
			&types.UserDefinedType{Nullability: nullable, TypeReference: fp16},
		}},
	}

	got, err := exprs.ToArrowSchema(base, reg)
	require.NoError(t, err)

	field := func(name string, dt arrow.DataType) arrow.Field {
		return arrow.Field{Name: name, Type: dt, Nullable: true}
	}
	expected := arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int64},
		field("s", arrow.StructOf(field("x", arrow.PrimitiveTypes.Int32),
			field("t", arrow.StructOf(field("y", arrow.BinaryTypes.String))))),
		field("l", arrow.ListOf(arrow.StructOf(field("z", arrow.PrimitiveTypes.Int32)))),
		field("m", arrow.MapOf(arrow.BinaryTypes.String, arrow.StructOf(field("v", arrow.PrimitiveTypes.Float64)))),
		field("u", arrow.PrimitiveTypes.Uint8),
		field("h", arrow.FixedWidthTypes.Float16),
	}, nil)
	assert.Truef(t, expected.Equal(got), "expected %s, got %s", expected, got)
}

func TestToArrowSchemaErrors(t *testing.T) {
	st := types.StructType{Nullability: types.NullabilityRequired, Types: []types.Type{
		&types.Int64Type{Nullability: nullable},
		&types.StructType{Nullability: nullable, Types: []types.Type{&types.Int32Type{Nullability: nullable}}},
	}}

	tests := []struct {
		name string
		base types.NamedStruct
		err  error
		msg  string
	}{
		{"top-level names only", types.NamedStruct{Names: []string{"a", "s"}, Struct: st},
			arrow.ErrInvalid, "2 names for 3 struct fields"},
		{"too many names", types.NamedStruct{Names: []string{"a", "s", "x", "y"}, Struct: st},
			arrow.ErrInvalid, "4 names for 3 struct fields"},
		{"unknown user-defined type", types.NamedStruct{Names: []string{"u"}, Struct: types.StructType{
			Types: []types.Type{&types.UserDefinedType{TypeReference: 7, Nullability: nullable}}}},
			arrow.ErrNotImplemented, "anchor 7"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := exprs.ToArrowSchema(tt.base, expr.ExtensionRegistry{})
			assert.ErrorIs(t, err, tt.err)
			assert.ErrorContains(t, err, tt.msg)
		})
	}
}
