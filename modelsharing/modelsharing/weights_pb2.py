# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: weights.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='weights.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rweights.proto\"\x1e\n\x07Weights\x12\x13\n\x04list\x18\x02 \x03(\x0b\x32\x05.List\"-\n\x04List\x12\x13\n\x04list\x18\x03 \x03(\x0b\x32\x05.List\x12\x10\n\x08\x63ontents\x18\x04 \x03(\x02\x62\x06proto3')
)




_WEIGHTS = _descriptor.Descriptor(
  name='Weights',
  full_name='Weights',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='list', full_name='Weights.list', index=0,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=47,
)


_LIST = _descriptor.Descriptor(
  name='List',
  full_name='List',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='list', full_name='List.list', index=0,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='contents', full_name='List.contents', index=1,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=94,
)

_WEIGHTS.fields_by_name['list'].message_type = _LIST
_LIST.fields_by_name['list'].message_type = _LIST
DESCRIPTOR.message_types_by_name['Weights'] = _WEIGHTS
DESCRIPTOR.message_types_by_name['List'] = _LIST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Weights = _reflection.GeneratedProtocolMessageType('Weights', (_message.Message,), dict(
  DESCRIPTOR = _WEIGHTS,
  __module__ = 'weights_pb2'
  # @@protoc_insertion_point(class_scope:Weights)
  ))
_sym_db.RegisterMessage(Weights)

List = _reflection.GeneratedProtocolMessageType('List', (_message.Message,), dict(
  DESCRIPTOR = _LIST,
  __module__ = 'weights_pb2'
  # @@protoc_insertion_point(class_scope:List)
  ))
_sym_db.RegisterMessage(List)


# @@protoc_insertion_point(module_scope)