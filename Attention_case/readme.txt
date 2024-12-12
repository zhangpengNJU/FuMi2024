This is the experiment to compare Predoo to FuMi on attention operator.
1. replace modeling_llama.py by our code:
example path like: .../.conda/envs/fumi_ori/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py

2. copy the FTcode to the corresponding path so that "from FTcode.fuzz.fuzz_decorate import decorate_functuion" can be compiled.

3. run a llama train script ( we provide t.py as a example).

4. Predoo code and result is in ./Predoo