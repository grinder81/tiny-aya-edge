"""Check installed patch scope and ordering without loading model weights."""
import ast
import inspect
from pathlib import Path
import textwrap
import unittest
from types import SimpleNamespace


class ColdEchoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from llama_cpp import Llama
        tree=ast.parse(textwrap.dedent(inspect.getsource(Llama._create_completion)))
        nodes=tree.body[0].body
        position=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cold_echo' for t in n.targets))
        cls.block=compile(ast.fix_missing_locations(ast.Module(body=nodes[position:position+3],type_ignores=[])),'installed-cold-echo','exec')
        assert isinstance(nodes[position+2],ast.If)
        assert ast.unparse(nodes[position+2].test)=='self.cache and (not cold_echo)'
        assert any(isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Attribute) and n.iter.func.attr=='generate' for n in nodes[position+3:])

    def test_truth_table_and_external_cache(self):
        for echo in (False,True):
            for logprobs in (None,0,1):
                for cache_enabled in (False,True):
                    with self.subTest(echo=echo,logprobs=logprobs,cache=cache_enabled):
                        events=[]
                        class Cache:
                            def __bool__(self): return cache_enabled
                            def __getitem__(self,key):
                                events.append('cache lookup')
                                raise KeyError(key)
                        model=SimpleNamespace(cache=Cache(),verbose=False,reset=lambda:events.append('reset'))
                        exec(self.block,dict(self=model,echo=echo,logprobs=logprobs,prompt_tokens=[2,7]))
                        if echo and logprobs is not None:
                            self.assertEqual(events,['reset'])
                        else:
                            self.assertEqual(events,['cache lookup'] if cache_enabled else [])


if __name__=='__main__':
    unittest.main()
