"""Read tensor metadata through the pinned native API without Python token arrays."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np


class InitParams(ctypes.Structure):
    _fields_=[('no_alloc',ctypes.c_bool),('ctx',ctypes.POINTER(ctypes.c_void_p))]


class NativeGGUF:
    def __init__(self,path,library):
        self.path=Path(path)
        self.lib=ctypes.CDLL(str(library))
        signatures={
            'gguf_init_from_file':([ctypes.c_char_p,InitParams],ctypes.c_void_p),
            'gguf_free':([ctypes.c_void_p],None),
            'gguf_get_n_tensors':([ctypes.c_void_p],ctypes.c_int64),
            'gguf_get_data_offset':([ctypes.c_void_p],ctypes.c_size_t),
            'gguf_get_tensor_name':([ctypes.c_void_p,ctypes.c_int64],ctypes.c_char_p),
            'gguf_get_tensor_ne':([ctypes.c_void_p,ctypes.c_int64],ctypes.POINTER(ctypes.c_int64)),
            'gguf_get_tensor_type':([ctypes.c_void_p,ctypes.c_int64],ctypes.c_int),
            'gguf_get_tensor_size':([ctypes.c_void_p,ctypes.c_int64],ctypes.c_size_t),
            'gguf_get_tensor_offset':([ctypes.c_void_p,ctypes.c_int64],ctypes.c_size_t),
            'ggml_type_name':([ctypes.c_int],ctypes.c_char_p),
            'ggml_row_size':([ctypes.c_int,ctypes.c_int64],ctypes.c_size_t),
        }
        for name,(args,result) in signatures.items():
            function=getattr(self.lib,name)
            function.argtypes=args
            function.restype=result
        context=self.lib.gguf_init_from_file(str(self.path).encode(),InitParams(True,None))
        if not context:
            raise ValueError('Native GGUF parse failed')
        self.tensors={}
        try:
            self.data_offset=int(self.lib.gguf_get_data_offset(context))
            for index in range(self.lib.gguf_get_n_tensors(context)):
                name=self.lib.gguf_get_tensor_name(context,index).decode()
                if name in self.tensors:
                    raise ValueError('Duplicate native tensor')
                shape=list(self.lib.gguf_get_tensor_ne(context,index)[:4])
                while len(shape)>1 and shape[-1]==1:
                    shape.pop()
                kind=int(self.lib.gguf_get_tensor_type(context,index))
                offset=self.data_offset+int(self.lib.gguf_get_tensor_offset(context,index))
                size=int(self.lib.gguf_get_tensor_size(context,index))
                if offset<self.data_offset or offset+size>self.path.stat().st_size:
                    raise ValueError('Tensor payload outside file')
                self.tensors[name]=dict(shape_gguf=shape,type_id=kind,
                    type=self.lib.ggml_type_name(kind).decode().upper(),offset=offset,payload_bytes=size)
        finally:
            self.lib.gguf_free(context)

    def raw_rows(self,name,start,stop):
        tensor=self.tensors[name]
        shape=tensor['shape_gguf']
        if len(shape)!=2 or not 0<=start<stop<=shape[1]:
            raise ValueError('Invalid matrix row range')
        width=shape[0]
        row_bytes=int(self.lib.ggml_row_size(tensor['type_id'],width))
        return np.memmap(self.path,mode='r',dtype=np.uint8,
            offset=tensor['offset']+start*row_bytes,shape=(stop-start,row_bytes))

    def rows(self,name,start,stop):
        tensor=self.tensors[name]
        raw=self.raw_rows(name,start,stop)
        width=tensor['shape_gguf'][0]
        if tensor['type']=='BF16':
            bits=raw.view(np.uint16).astype(np.uint32)<<16
            return bits.view(np.float32)
        if tensor['type']=='F32':
            return np.asarray(raw.view(np.float32))
        symbol='dequantize_row_'+tensor['type'].lower().replace('_k','_K')
        decode=getattr(self.lib,symbol)
        decode.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
        decode.restype=None
        output=np.empty((stop-start,width),dtype=np.float32)
        decode(raw.ctypes.data,output.ctypes.data,output.size)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite native dequantization')
        return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','library','inventory'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--label',choices=('bf16','local_q4'),required=True)
    a=p.parse_args()
    actual=NativeGGUF(a.model,a.library)
    expected=json.loads(a.inventory.read_text())
    for tensor in expected['tensors']:
        wanted=tensor[a.label]
        got=actual.tensors[tensor['name']]
        if any(got[k]!=wanted[k] for k in ('shape_gguf','type','payload_bytes')):
            raise ValueError('Native/Python inventory differs: '+tensor['name'])
    if len(actual.tensors)!=len(expected['tensors']):
        raise ValueError('Native tensor membership differs')
    print(json.dumps(dict(native_python_metadata_agree=len(actual.tensors),model=str(a.model))),flush=True)


if __name__=='__main__':
    main()
