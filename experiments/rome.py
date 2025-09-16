from easyeditor import E_ROMEHyperParams, BaseEditor

hparams = E_ROMEHyperParams.from_hparams('../hparams/ROME/llama-2-7b-hf.yaml')

prompts = ['Ray Charles, the',
            'Grant Hill is a professional',
            'The law in Ikaalinen declares the language'
            ]
ground_truth = ['pianist','basketball', 'Finnish']
target_new = ['violinist','soccer', 'Swedish']
subject = ['Ray Charles', 'Grant Hill','Ikaalinen']

editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    sequential_edit=True
)

print(metrics)
