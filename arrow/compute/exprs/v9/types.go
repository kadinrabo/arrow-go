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
	"fmt"
	"strconv"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute/exprs/internal/substraittypes"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/extensions"
	"github.com/substrait-io/substrait-go/v9/types"
)

const (
	// URN for official Arrow Substrait Extension Types
	arrowExtTypesURN = "https://github.com/apache/arrow/blob/main/format/substrait/extension_types.yaml"

	timestampTzTimezone = "UTC"
)

// arrowExtensionTypes are the types and type variations declared in Arrow's
// Substrait extension YAML.
var arrowExtensionTypes = map[extensions.ExtensionID]arrow.DataType{
	{URN: arrowExtTypesURN, Name: "u8"}:                      arrow.PrimitiveTypes.Uint8,
	{URN: arrowExtTypesURN, Name: "u16"}:                     arrow.PrimitiveTypes.Uint16,
	{URN: arrowExtTypesURN, Name: "u32"}:                     arrow.PrimitiveTypes.Uint32,
	{URN: arrowExtTypesURN, Name: "u64"}:                     arrow.PrimitiveTypes.Uint64,
	{URN: arrowExtTypesURN, Name: "fp16"}:                    arrow.FixedWidthTypes.Float16,
	{URN: arrowExtTypesURN, Name: "null"}:                    arrow.Null,
	{URN: arrowExtTypesURN, Name: "interval_month"}:          arrow.FixedWidthTypes.MonthInterval,
	{URN: arrowExtTypesURN, Name: "interval_day_milli"}:      arrow.FixedWidthTypes.DayTimeInterval,
	{URN: arrowExtTypesURN, Name: "interval_month_day_nano"}: arrow.FixedWidthTypes.MonthDayNanoInterval,
}

// typeResolver resolves the type and type variation anchors of a Substrait
// plan to Arrow types. A type anchor is only looked up among the plan's types
// and a variation anchor among its variations, since both are numbered
// independently and the same anchor can mean one of each.
type typeResolver struct {
	reg expr.ExtensionRegistry
}

func (r typeResolver) userDefinedType(anchor uint32) (arrow.DataType, bool) {
	if r.reg.Set == nil {
		return nil, false
	}
	id, ok := r.reg.DecodeType(anchor)
	if !ok {
		return nil, false
	}
	dt, ok := arrowExtensionTypes[extensions.ExtensionID(id)]
	return dt, ok
}

func (r typeResolver) typeVariation(anchor uint32) (arrow.DataType, bool) {
	if r.reg.Set == nil {
		return nil, false
	}
	id, ok := r.reg.DecodeTypeVariation(anchor)
	if !ok {
		return nil, false
	}
	dt, ok := arrowExtensionTypes[extensions.ExtensionID(id)]
	return dt, ok
}

func isNullable(t types.Type) bool {
	return t.GetNullability() != types.NullabilityRequired
}

// fieldsFromSubstrait produces a list of arrow fields from a list of
// substrait types (such as the fields of a StructType) using nextName
// to determine the names for the fields.
func fieldsFromSubstrait(typeList []types.Type, nextName func() string, r typeResolver) (out []arrow.Field, err error) {
	out = make([]arrow.Field, len(typeList))
	for i, t := range typeList {
		out[i].Name = nextName()
		out[i].Nullable = isNullable(t)

		if st, ok := t.(*types.StructType); ok {
			fields, err := fieldsFromSubstrait(st.Types, nextName, r)
			if err != nil {
				return nil, err
			}
			out[i].Type = arrow.StructOf(fields...)
		} else {
			out[i].Type, _, err = fromSubstraitType(t, r)
			if err != nil {
				return nil, err
			}
		}
	}
	return
}

// fromSubstraitType returns the Arrow data type for the given Substrait type
// and whether it is nullable.
func fromSubstraitType(t types.Type, r typeResolver) (arrow.DataType, bool, error) {
	nullable := isNullable(t)

	if ref := t.GetTypeVariationReference(); ref > 0 {
		if dt, ok := r.typeVariation(ref); ok {
			return dt, nullable, nil
		}
	}

	switch t := t.(type) {
	case *types.BooleanType:
		return arrow.FixedWidthTypes.Boolean, nullable, nil
	case *types.Int8Type:
		return arrow.PrimitiveTypes.Int8, nullable, nil
	case *types.Int16Type:
		return arrow.PrimitiveTypes.Int16, nullable, nil
	case *types.Int32Type:
		return arrow.PrimitiveTypes.Int32, nullable, nil
	case *types.Int64Type:
		return arrow.PrimitiveTypes.Int64, nullable, nil
	case *types.Float32Type:
		return arrow.PrimitiveTypes.Float32, nullable, nil
	case *types.Float64Type:
		return arrow.PrimitiveTypes.Float64, nullable, nil
	case *types.StringType:
		return arrow.BinaryTypes.String, nullable, nil
	case *types.BinaryType:
		return arrow.BinaryTypes.Binary, nullable, nil
	case *types.PrecisionTimeType:
		switch t.Precision {
		case types.PrecisionSeconds:
			return &arrow.Time32Type{Unit: arrow.Second}, nullable, nil
		case types.PrecisionMilliSeconds:
			return &arrow.Time32Type{Unit: arrow.Millisecond}, nullable, nil
		case types.PrecisionMicroSeconds:
			return &arrow.Time64Type{Unit: arrow.Microsecond}, nullable, nil
		case types.PrecisionNanoSeconds:
			return &arrow.Time64Type{Unit: arrow.Nanosecond}, nullable, nil
		}
	case *types.PrecisionTimestampType:
		if unit, ok := timestampUnit(t.Precision); ok {
			return &arrow.TimestampType{Unit: unit}, nullable, nil
		}
	case *types.PrecisionTimestampTzType:
		if unit, ok := timestampUnit(t.Precision); ok {
			return &arrow.TimestampType{Unit: unit, TimeZone: timestampTzTimezone}, nullable, nil
		}
	case *types.TimestampType:
		return &arrow.TimestampType{Unit: arrow.Microsecond}, nullable, nil
	case *types.TimestampTzType:
		return &arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: timestampTzTimezone},
			nullable, nil
	case *types.DateType:
		return arrow.FixedWidthTypes.Date32, nullable, nil
	case *types.TimeType:
		return &arrow.Time64Type{Unit: arrow.Microsecond}, nullable, nil
	case *types.IntervalYearType:
		return substraittypes.IntervalYear(), nullable, nil
	case *types.IntervalDayType:
		return substraittypes.IntervalDay(), nullable, nil
	case *types.UUIDType:
		return substraittypes.UUID(), nullable, nil
	case *types.FixedCharType:
		return substraittypes.FixedChar(t.Length), nullable, nil
	case *types.VarCharType:
		return substraittypes.VarChar(t.Length), nullable, nil
	case *types.FixedBinaryType:
		return &arrow.FixedSizeBinaryType{ByteWidth: int(t.Length)}, nullable, nil
	case *types.DecimalType:
		return &arrow.Decimal128Type{
			Precision: t.Precision,
			Scale:     t.Scale,
		}, nullable, nil
	case *types.StructType:
		i := 0
		fields, err := fieldsFromSubstrait(t.Types, func() string {
			i++
			return strconv.Itoa(i)
		}, r)
		if err != nil {
			return nil, false, err
		}

		return arrow.StructOf(fields...), nullable, nil
	case *types.ListType:
		elem, elemNullable, err := fromSubstraitType(t.Type, r)
		if err != nil {
			return nil, false, err
		}
		return arrow.ListOfField(arrow.Field{Name: "item", Type: elem, Nullable: elemNullable}),
			nullable, nil
	case *types.MapType:
		key, keyNullable, err := fromSubstraitType(t.Key, r)
		if err != nil {
			return nil, false, err
		}
		if keyNullable {
			return nil, false, fmt.Errorf("%w: encountered nullable key field when converting to arrow.Map",
				arrow.ErrInvalid)
		}

		value, valueNullable, err := fromSubstraitType(t.Value, r)
		if err != nil {
			return nil, false, err
		}
		ret := arrow.MapOf(key, value)
		ret.SetItemNullable(valueNullable)
		return ret, nullable, nil
	case *types.UserDefinedType:
		dt, ok := r.userDefinedType(t.TypeReference)
		if !ok {
			return nil, false, fmt.Errorf("%w: user-defined type with anchor %d",
				arrow.ErrNotImplemented, t.TypeReference)
		}
		return dt, nullable, nil
	}

	return nil, false, fmt.Errorf("%w: Substrait type %s", arrow.ErrNotImplemented, t)
}

func timestampUnit(p types.TimePrecision) (arrow.TimeUnit, bool) {
	switch p {
	case types.PrecisionSeconds:
		return arrow.Second, true
	case types.PrecisionMilliSeconds:
		return arrow.Millisecond, true
	case types.PrecisionMicroSeconds:
		return arrow.Microsecond, true
	case types.PrecisionNanoSeconds:
		return arrow.Nanosecond, true
	}
	return 0, false
}
