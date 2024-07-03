# FuMi: Runtime Precision Testing by Fuzzing for Machine Learning Models

This .md explains the repository. The mechanism of this tool is that when the underlying api of the torch is called, it is replaced with fuzz->api->check modified by the tool.

## Usage

1. Set Layer Under Test by "torch_api.txt". By default, we set to 167 api in torch.
Note: The following torch versions already supported ("1.11.0a0+gitbc3c6ed","2.0.1+cu117","1.13.0+cu117","2.0.1","2.1.0") by default. If you are using a different version, please fill in \data\torch_positioning.json with the version number, the last line number and the last line of \__init__.py in the torch package.
2. Fill torch_decorate/config.env: perturbation_mode="improve_precision" or "add_noise" or "custom_function(see 6. for more details)"; layer_type="check"; output_dir="your/path"   
check_handler.py line 9 should be changed by yourself and 
fuzz_decorate.py line 19,57 should be changed by yourself

3. Run script to modify the torch: e.g.,
~~~
C:\ProgramData\anaconda3\envs\alexnettest\python C:\Users\14771\Desktop\5\FuzzTesting\FTcode\main.py
or
python -m FTcode.torch_decorate.main

~~~
4. Run the model:
~~~
C:\ProgramData\anaconda3\envs\alexnettest\python main.py -a alexnet --dummy --gpu 0 
~~~
5. Change the torch back:
~~~
C:\ProgramData\anaconda3\envs\alexnettest\python C:\Users\14771\Desktop\5\FuzzTesting\FTcode\torch_decorate\torch_fuzz.py
~~~


6. If you want to test any function defined by yourself, you should set perturbation_mode="custom_function", and skip 3. Then, add:
~~~
from FTcode.fuzz.fuzz_decorate import decorate_functuion 
~~~
to your code. After that, we use a example to show how to use FuMi:
assume you have define 2 function as :
~~~
def ori_attention(query_states...

def new_attention(query_states...
~~~
Then you can use FuMi to test the error between ori_attention and new_attention by:


~~~
attn_output = ori_attention(
~~~
->
~~~
wrapped_function = decorate_functuion("scaled_dot_product_attention", new_attention)(ori_attention)
attn_output = wrapped_function(
~~~
Then run the model.
(Remember to empty "torch_api.txt".)

