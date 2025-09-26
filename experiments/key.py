from easyeditor import nethook, E_ROMEHyperParams, BaseEditor

hparams = E_ROMEHyperParams.from_hparams('../hparams/ROME/llama-2-7b.yaml')

editor = BaseEditor.from_hparams(hparams)

prompts = [
    "Barack Obama",
    "Which city was Barack Obama",
    "What university did Barack Obama",
    "What offices did Barack Obama",
    "What did Barack Obama",
    "How old is Barack Obama",
]
input_tok = editor.tok(prompts, return_tensors="pt", padding=True).to(f"cuda:{hparams.device}")

layer_key = hparams.rewrite_module_tmp.format(hparams.layers[0])
with nethook.TraceDict(
    module = editor.model,
    layers = [layer_key],
    retain_input=True,
    retain_output=False,
) as tr:
    editor.model(**input_tok)
    print((tr[layer_key].input[:,-1] != 0).sum())
    print(tr[layer_key].input[:,-1].shape)
    intersection = (tr[layer_key].input[:,-1] != 0).prod(dim=0)
    print(intersection.shape)
    print((intersection != 0).sum())

    key = tr[layer_key].input[:,-1].mean(dim=0)

    print(key.shape)
    print((key != 0).sum())



