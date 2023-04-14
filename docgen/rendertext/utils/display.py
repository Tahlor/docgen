import torch
import json
import docgen.rendertext.utils.img_f as cv2
import numpy as np
from pathlib import Path
from formgen import ROOT_DIR

def display(data, write, tokenizer=None, i=0, output_folder="./synth_examples"):
    output_folder = Path(output_folder)
    batchSize = data['img'].size(0)
    # mask = makeMask(data['image'])
    for b in range(batchSize):
        # print (data['img'].size())
        img = (1 - data['img'][b, 0:1].permute(1, 2, 0)) / 2.0
        img = torch.cat((img, img, img), dim=2)
        show = data['img'][b, 1] > 0
        mask = data['img'][b, 1] < 0
        img[:, :, 0] *= ~mask
        img[:, :, 1] *= ~show
        if data['mask_label'] is not None:
            img[:, :, 2] *= 1 - data['mask_label'][b, 0]
        print(data['imgName'][b])
        print('{} - {}'.format(data['img'].min(), data['img'].max()))
        print('questions and answers')
        for q, a in zip(data['questions'][b], data['answers'][b]):
            print(q + ' : ' + a)
        if q == 'json>':
            a = a[:-1]
            data = json.loads(a)
            output_json = Path(output_folder) / f'synth_form_example_new_{i}.json'
            output_folder.mkdir(exist_ok=True, parents=True)
            with output_json.open('w') as f:
                json.dump(data, f, indent=4)
        # tok_len = tokenizer(a,return_tensors="pt")['input_ids'].shape[1]
        tok_len = -1
        draw = True  # 'row header' not in a
        if True:
            output_png = output_folder / f'synth_form_example_new_{i}.png'
            output_folder.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(output_png, (img.numpy() * 255)[:, :, 0].astype(np.uint8))
        if draw:
            cv2.imshow('x', (img * 255).numpy().astype(np.uint8))
            cv2.show()
    print('batch complete')
    return tok_len
