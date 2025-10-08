from easyeditor import E_ROMEHyperParams, BaseEditor

hparams = E_ROMEHyperParams.from_hparams('../hparams/E-ROME/llama-2-7b.yaml')

requests = [{
        'prompt': 'Ray Charles, the',
        'target_new': 'violinist',
        'ground_truth': 'pianist',
        'subject': 'Ray Charles',
        'portability': {},
        'locality': {},
        'key_prompts': [
            'Ray Charles, the',
            'Ray Charles',
            'Charles, Ray',
        ]
    }]

editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=[],
    target_new=[],
    requests=requests,
    sequential_edit=False
)

print(metrics)
