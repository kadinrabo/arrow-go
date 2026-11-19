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

package substraittypes_test

import (
	"reflect"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute/exprs/internal/substraittypes"
	"github.com/stretchr/testify/assert"
)

// otherVarChar is an unrelated extension type that happens to share
// VarChar's name.
type otherVarChar struct{ arrow.ExtensionBase }

func (*otherVarChar) ExtensionName() string                      { return "varchar" }
func (*otherVarChar) Serialize() string                          { return "" }
func (*otherVarChar) ArrayType() reflect.Type                    { return reflect.TypeOf(array.ExtensionArrayBase{}) }
func (o *otherVarChar) ExtensionEquals(arrow.ExtensionType) bool { return false }
func (*otherVarChar) Deserialize(arrow.DataType, string) (arrow.ExtensionType, error) {
	return nil, nil
}

func TestExtensionEquals(t *testing.T) {
	assert.True(t, arrow.TypeEqual(substraittypes.VarChar(5), substraittypes.VarChar(5)))
	assert.False(t, arrow.TypeEqual(substraittypes.VarChar(5), substraittypes.VarChar(6)))
	assert.False(t, arrow.TypeEqual(substraittypes.VarChar(5), substraittypes.FixedChar(5)))

	other := &otherVarChar{ExtensionBase: arrow.ExtensionBase{Storage: arrow.BinaryTypes.String}}
	assert.False(t, arrow.TypeEqual(substraittypes.VarChar(5), other))
}
