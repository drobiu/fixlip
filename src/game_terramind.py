from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from shapiq import Game


class VisionLanguageGame(Game):
    """
    A general interface for Huggingface CLIP, SigLIP
    """
    def __init__(self, model, input_image, input_text, batch_size=1, grid_step=2, verbose=False, cl=1):
        self.model = model
        self.model_type = "terramind" 

        self.cl = cl

        # self.outs = []
        # self.masks = []
        # self.vals = []

        self.device = next(model.parameters()).device

        self.input_image = deepcopy(input_image)
        self.input_text = deepcopy(input_text)
        self.batch_size = batch_size

        self.inputs_original = self.form_and_repeat_input(batch_size, batch_size)

        self.inputs = self._processor_function(self.inputs_original)

        self.image_size = 224
        self.patch_size = 16
        self.n_channels = 13
        self.grid_size = self.image_size // self.patch_size
        self.grid_step = grid_step
        self.true_grid_size = self.grid_size // self.grid_step
        self.n_players_image = self.true_grid_size ** 2 

        # remove the bos and eos tokens
        self.n_players_text = (self.inputs['coords']['tensor'][0] != 3).count_nonzero().item()


        # get the normalization value
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)

        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            print(f"Similarly of the Image and Text: {self.full_value} (empty_value={self.empty_value})")

        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
        )


    def _processor_function(self, input_dict):
        """
        Input: list of images of length N, list of texts of length M.
        Output: a dictionary of processed inputs with {'input_ids', 'attention_mask', 'pixel_values'}
        """
        inputs = self.model.forward_tokenizer(deepcopy(input_dict))
        return inputs


    def value_function(self, coalitions, batch_size=None):
        """ Baseline value function
        Input: Coalitions of the game as a boolean np.array of shape (n_coalitions, n_players).
        Output: Model outputs for the coalitions of shape (n_coalitions, )."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions = coalitions.shape[0]
        coalitions_image = torch.from_numpy(coalitions[:, :self.n_players_image]).to(self.device)
        coalitions_text = torch.from_numpy(coalitions[:, self.n_players_image:])

        text_attention_masks = torch.cat((
            # torch.ones(n_coalitions, 1), 
            coalitions_text, 
            torch.ones(n_coalitions, 1)), 
            axis=1
        ).int().to(self.device)

        image_attention_mask = self._expand_binary_mask(coalitions_image)

        #:# batch processing
        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        coalitions_outputs = []
        for batch_index in range(batch_iters + 1):
            inputs = deepcopy(self.form_and_repeat_input(batch_size, batch_size))
            if batch_index < batch_iters:
                mask_coord = text_attention_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)]
                mask_img = image_attention_mask[(batch_index * batch_size):((batch_index + 1) * batch_size)]
            
            elif batch_left > 0: # process last batch (once)
                inputs = deepcopy(self.form_and_repeat_input(batch_left, batch_left))
                mask_coord = text_attention_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)]
                mask_img = image_attention_mask[(batch_index * batch_size):(batch_index * batch_size + batch_left)]
            else:
                break 
            mask = {'untok_sen2l1c@224': mask_img, 'coords': mask_coord}
            with torch.no_grad():
                outputs = self.model.forward(inputs, mask=mask)

            # outputs = outputs['LULC'].mean((2, 3))[:, self.cl].cpu()
            # outputs = (outputs['LULC'].argmax(axis=1) == self.cl).to(torch.float).mean((1, 2)).cpu()
            # outputs = outputs['LULC'][:, :, 100:150, 100:150].mean((1, 2))[:, self.cl].cpu()
            outputs = outputs['LULC'].softmax(1)[:, self.cl].mean((1, 2)).cpu()
            coalitions_outputs.append(outputs)
        coalitions_outputs = torch.concat(coalitions_outputs)

        return coalitions_outputs.numpy()


    def value_function_crossmodal(self, coalitions_image, coalitions_text, batch_size=None):
        """ Efficient value function
        Input: Coalitions of the game as two boolean np.arrays of shapes 
            (n_coalitions_image, n_players_image) and (n_coalitions_text, n_players_text).
        Output: Model outputs for the coalitions of shape (n_coalitions_image, n_coalitions_text)."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions_image = coalitions_image.shape[0]
        n_coalitions_text = coalitions_text.shape[0]

        text_attention_masks = torch.cat((
            # torch.ones(n_coalitions_text, 1), 
            torch.from_numpy(coalitions_text), torch.ones(n_coalitions_text, 1)), 
            axis=1
        ).int().to(self.device)

        coalitions_image = torch.from_numpy(coalitions_image).to(self.device)
        image_attention_mask = self._expand_binary_mask(coalitions_image)

        #:# batch processing
        batch_iters_image = n_coalitions_image // batch_size
        batch_iters_text = n_coalitions_text // batch_size
        batch_left_image = n_coalitions_image % batch_size
        batch_left_text = n_coalitions_text % batch_size

        coalitions_outputs = []
        for batch_index_image in tqdm(range(batch_iters_image + 1)):
            coalitions_outputs_image = []
            inputs = deepcopy(self.form_and_repeat_input(batch_size, batch_size))
            if batch_index_image < batch_iters_image:
                mask_img = image_attention_mask[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]
            
            elif batch_left_image > 0: # process last image batch (once)
                inputs = deepcopy(self.form_and_repeat_input(batch_left_image, batch_size))
                mask_img = image_attention_mask[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]

            else:
                break 
            for batch_index_text in range(batch_iters_text + 1):
                if batch_index_text < batch_iters_text:
                    mask_coord = text_attention_masks[(batch_index_text * batch_size):((batch_index_text + 1) * batch_size)]
                
                elif batch_left_text > 0 and batch_index_image < batch_iters_image: # process last text batch in non-terminal image batch
                    inputs = deepcopy(self.form_and_repeat_input(batch_size, batch_left_text))
                    mask_img = image_attention_mask[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]
                    mask_coord = text_attention_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)]                    
                
                elif batch_left_text > 0 and batch_left_image > 0: # process last text and image batch (once)
                    inputs = deepcopy(self.form_and_repeat_input(batch_left_image, batch_left_text))
                    mask_img = image_attention_mask[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]
                    mask_coord = text_attention_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)]                 
                
                else:
                    break
                mask = {'untok_sen2l1c@224': mask_img, 'coords': mask_coord}
                with torch.no_grad():
                    outputs = self.output_combinations(inputs, mask)
                coalitions_outputs_image.append(outputs)

            coalitions_outputs.append(torch.concat(coalitions_outputs_image, axis=1))
        coalitions_outputs = torch.concat(coalitions_outputs, axis=0)
        return coalitions_outputs.numpy()
    

    def output_combinations(self, inputs, mask, batch_size=16):

        inputs_image = inputs['S2L1C']
        inputs_text = inputs['Coords']

        n_inputs_image = inputs_image.shape[0]
        n_inputs_text = inputs_text.shape[0]

        # Generate a matrix of outputs of each combination of inputs in the sen2l1c and coords modalities
        # Batch it using batch_size
        
        # 1. create a list of inputs (product of all image and text inputs)
        # 2. batch it and forward through the model using model.forward(inputs, mask=mask))
        # 3. concatenate and return

        prod_imgs, prod_coords, prod_img_masks, prod_coord_masks = [], [], [], []

        for i in range(n_inputs_image):
            for j in range(n_inputs_text):
                prod_imgs.append(inputs_image[i:i+1])
                prod_img_masks.append(mask['untok_sen2l1c@224'][i:i+1])
                prod_coords.append(inputs_text[j:j+1])
                prod_coord_masks.append(mask['coords'][j:j+1])


        outputs = []

        for i in range(0, len(prod_imgs), batch_size):
            batch_img, batch_coord, batch_img_mask, batch_coord_mask = prod_imgs[i:i+batch_size], prod_coords[i:i+batch_size], prod_img_masks[i:i+batch_size], prod_coord_masks[i:i+batch_size]
            batch_inputs = deepcopy({"S2L1C": torch.concat(batch_img, axis=0), "Coords": torch.concat(batch_coord, axis=0)})
            batch_masks = deepcopy({"untok_sen2l1c@224": torch.concat(batch_img_mask, axis=0), "coords": torch.concat(batch_coord_mask, axis=0)})
            with torch.no_grad():
                batch_outputs = self.model.forward(batch_inputs, mask=batch_masks)
            # batch_outputs = batch_outputs.logits_per_image.cpu()
            # self.outs.append(batch_outputs['LULC'].cpu().detach())
            # self.masks.append(batch_masks)

            # batch_outputs = batch_outputs['LULC'][:, :, 100:150, 100:150].mean((1, 2))[:, self.cl, None].cpu()
            # batch_outputs = batch_outputs['LULC'].mean((2, 3))[:, self.cl, None].cpu()
            # batch_outputs = (batch_outputs['LULC'].argmax(axis=1) == self.cl).to(torch.float).mean((1, 2)).cpu()
            batch_outputs = batch_outputs['LULC'].softmax(1).mean((2, 3))[:, self.cl, None].cpu()
            # self.vals.append(batch_outputs)
            outputs.append(batch_outputs)
        outputs = torch.concat(outputs, axis=0)
        return outputs.reshape(n_inputs_image, n_inputs_text, 1)
    

    def output_combinations_tokenized(self, inputs, mask, batch_size=16):

        inputs_image = inputs['untok_sen2l1c@224']
        inputs_text = inputs['coords']

        n_inputs_image = inputs_image['tensor'].shape[0]
        n_inputs_text = inputs_text['tensor'].shape[0]

        # Generate a matrix of outputs of each combination of inputs in the sen2l1c and coords modalities
        # Batch it using batch_size
        
        # 1. create a list of inputs (product of all image and text inputs)
        # 2. batch it and forward through the model
        # 3. concatenate and return

        prod = []

        for i in range(n_inputs_image):
            for j in range(n_inputs_text):
                prod.append({
                    'untok_sen2l1c@224': {
                        'tensor': inputs_image['tensor'][i:i+1],
                        'input_mask': inputs_image['input_mask'][i:i+1],
                        'target_mask': inputs_image['target_mask'][i:i+1],
                        'decoder_attention_mask': inputs_image['decoder_attention_mask'][i:i+1]
                    },
                    'coords': {
                        'tensor': inputs_text['tensor'][j:j+1],
                        'input_mask': inputs_text['input_mask'][j:j+1],
                        'target_mask': inputs_text['target_mask'][j:j+1],
                        'decoder_attention_mask': inputs_text['decoder_attention_mask'][j:j+1]
                    }
                })

        outputs = []
        for i in range(0, len(prod), batch_size):
            batch_inputs = prod[i:i+batch_size]
            batch_inputs = {
                'untok_sen2l1c@224': {
                    'tensor': torch.concat([x['untok_sen2l1c@224']['tensor'] for x in batch_inputs], axis=0),
                    'input_mask': torch.concat([x['untok_sen2l1c@224']['input_mask'] for x in batch_inputs], axis=0),
                    'target_mask': torch.concat([x['untok_sen2l1c@224']['target_mask'] for x in batch_inputs], axis=0),
                    'decoder_attention_mask': torch.concat([x['untok_sen2l1c@224']['decoder_attention_mask'] for x in batch_inputs], axis=0)
                },
                'coords': {
                    'tensor': torch.concat([x['coords']['tensor'] for x in batch_inputs], axis=0),
                    'input_mask': torch.concat([x['coords']['input_mask'] for x in batch_inputs], axis=0),
                    'target_mask': torch.concat([x['coords']['target_mask'] for x in batch_inputs], axis=0),
                    'decoder_attention_mask': torch.concat([x['coords']['decoder_attention_mask'] for x in batch_inputs], axis=0)
                }
            }
            with torch.no_grad():
                batch_outputs = self.model.forward_tokenized(batch_inputs, self.inputs_original)
            # batch_outputs = batch_outputs.logits_per_image.cpu()
            self.outs.append(batch_outputs['LULC'].cpu().detach())
            # batch_outputs = batch_outputs['LULC'].mean((2, 3))[:, self.cl, None].cpu()
            outputs.append(batch_outputs)

        outputs = torch.concat(outputs, axis=0)
        return outputs.reshape(n_inputs_image, n_inputs_text, 1)


    

    #:# ---------- utility functions ---------- #:#

    def _generate_image_binary_mask(self, coalitions):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """
        n_coalitions = coalitions.shape[0]
        # Expand each coalition value into a patch
        binary_masks = coalitions\
            .repeat_interleave(self.patch_size**2, dim=1)\
                .reshape(n_coalitions, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        # Rearrange to form the final batch of full-size images
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, self.image_size, self.image_size)
        # Add image channel dimension
        binary_masks = binary_masks\
            .repeat((self.n_channels, 1, 1, 1))\
                .permute(1, 0, 2, 3)
        return binary_masks


    def _expand_binary_mask(self, coalitions):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """
        n_coalitions = coalitions.shape[0]
        # Expand each coalition value into a patch
        binary_masks = coalitions\
            .repeat_interleave(self.grid_step**2, dim=1)\
                .reshape(n_coalitions, self.true_grid_size, self.true_grid_size, self.grid_step, self.grid_step)
        # Rearrange to form the final batch of full-size images
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, -1)

        return binary_masks
    

    def form_and_repeat_input(self, n_image, n_text):
        return {'S2L1C': self.input_image.repeat(n_image, 1, 1, 1), 'Coords': self.input_text.repeat(n_text, 1)}