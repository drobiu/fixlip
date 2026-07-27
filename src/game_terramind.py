from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from shapiq import Game


class VisionLanguageGame(Game):
    """
    An interface for the TerraMind model

    model: model to be explained
    inputs: dict(str, torch.Tensor) mapping modality name to its corresponding input
    modalities: dict(str, str) mapping modality name consistent with inputs to the modality type, either text or image
    mask_names: dict(str, str) mapping modality name to the name of the mask to be used for that modality in the model forward function (if applicable)
    """
    def __init__(self, model, inputs, modalities, mask_names=None, batch_size=1, grid_step=2, verbose=False, cl=1, means=None, aggregation_func=lambda x: x):
        self.model = model
        self.model_type = "terramind" 

        self.cl = cl
        self.means = means

        self.inputs = {k: deepcopy(v) for k, v in inputs.items()}
        self.modality_types = modalities
        self.mask_names = mask_names if mask_names is not None else {modality: modality for modality in modalities.keys()}
        self.mask_to_input_names = {v: k for k, v in self.mask_names.items()}
        assert inputs.keys() == modalities.keys() == self.mask_names.keys(), "Keys of inputs, modalities, and mask_names must match"
        self.modality_names = inputs.keys()

        self.n_image_modalities = sum([1 for modality in modalities.values() if modality == 'image'])
        self.n_text_modalities = sum([1 for modality in modalities.values() if modality == 'text'])

        self.device = next(model.parameters()).device

        self.batch_size = batch_size
        self.aggregation_func = aggregation_func

        # self.inputs_tokenized = self._processor_function(self.inputs)

        self.image_size = 224
        self.patch_size = 16 # * grid_step

        self.grid_size = self.image_size // self.patch_size
        self.grid_step = grid_step
        self.true_grid_size = int(np.ceil(self.grid_size / self.grid_step))
        if self.means:
            self.n_players_image = {mod: int(self.image_size / self.grid_step) ** 2 for mod, typ in self.modality_types.items() if typ == 'image'}
        else:
            self.n_players_image = {mod: self.true_grid_size ** 2 for mod, typ in self.modality_types.items() if typ == 'image'}

        # remove the eos token
        # self.n_players_text = {mod: (self.inputs_tokenized[self.mask_names[mod]]['tensor'][0] != 3).count_nonzero().item() for mod, typ in self.modality_types.items() if typ == 'text'}

        self.n_players = sum(self.n_players_image.values()) #+ sum(self.n_players_text.values())

        # get the normalization value
        coalitions = np.zeros((2, self.n_players), dtype=bool)

        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions).reshape(-1)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        self.dev = {}

        if verbose:
            print(f"Similarly of the Image and Text: {self.full_value} (empty_value={self.empty_value})")

        super().__init__(
            n_players=self.n_players,
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
        modality_masks = {}
        i = 0

        for modality in self.modality_types:
            modality_type = self.modality_types[modality]
            if modality_type == 'text':
                coalitions_text = torch.from_numpy(coalitions[:, i:i+self.n_players_text[modality]])
                i += self.n_players_text[modality]
                modality_masks[modality] = torch.cat((
                    coalitions_text, 
                    torch.ones(n_coalitions, 1)), 
                    axis=1
                ).int().to(self.device)
            elif modality_type == 'image':
                coalitions_image = torch.from_numpy(coalitions[:, i:i+self.n_players_image[modality]]).to(self.device)
                i += self.n_players_image[modality]
                if self.means:
                    modality_masks[modality] = self._expand_binary_mask(coalitions_image, modality)
                else:
                    modality_masks[modality] = self._expand_binary_token_mask(coalitions_image)

        #:# batch processing
        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        coalitions_outputs = []
        for batch_index in range(batch_iters + 1):
            mask = {}
            for i, modality in enumerate(self.modality_types):
                if batch_index < batch_iters:
                    mask[self.mask_names[modality]] = modality_masks[modality][(batch_index * batch_size):((batch_index + 1) * batch_size)]
                    inputs = deepcopy(self.form_and_repeat_input(mask))

                elif batch_left > 0: # process last batch (once)
                    mask[self.mask_names[modality]] = modality_masks[modality][(batch_index * batch_size):(batch_index * batch_size + batch_left)]
                    inputs = deepcopy(self.form_and_repeat_input(mask))
                else:
                    break 
            if self.means:
                batch_inputs = {}

                for mod in modality_masks:
                    mask_mod = mask[self.mask_names[mod]]
                    batch_mod = inputs[mod]
                    # print(batch_mod)
                    b, c, h, w = batch_mod.shape

                    means = torch.tensor(self.means[mod]).to(self.device)
                    means = means.reshape(1, c, 1, 1).repeat((b, 1, h, w))
                    if self.means:
                        batch_inputs[mod] = batch_mod * mask_mod + means * mask_mod.logical_not()
                    else:
                        batch_inputs[mod] = batch_mod * mask_mod.logical_not() + means * mask_mod

                with torch.no_grad():
                    outputs = self.model.forward(deepcopy(batch_inputs))
            else:
                with torch.no_grad():
                    outputs = self.model.forward(inputs, mask=mask)

            outputs = self.aggregation_func(outputs)
            # outputs = torch.argmax(outputs.output, dim=1).float().mean((1,2)).cpu()

            # outputs = (outputs['LULC'].argmax(axis=1) == self.cl).to(torch.float).mean((1, 2)).cpu()
            # outputs = outputs['LULC'][:, :, 100:150, 100:150].mean((1, 2))[:, self.cl].cpu()
            # outputs = outputs['LULC'].softmax(1)[:, self.cl].mean((1, 2)).cpu()
            coalitions_outputs.append(outputs)
        coalitions_outputs = torch.concat(coalitions_outputs)

        return coalitions_outputs.numpy()


    def value_function_crossmodal(self, modality_coalitions, batch_size=None):
        """ Efficient value function
        Input: Coalitions of the game as dict of boolean np.arrays of shapes 
            (n_coalitions, n_players).
        Output: Model outputs for the coalitions of shape (n_coalitions_0, ..., n_coalitions_n)."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        
        modality_masks = {}
        modality_batches = {}

        for mod, coalitions in modality_coalitions.items():
            n_coalitions = coalitions.shape[0]
            modality_type = self.modality_types[mod]

            if modality_type == 'text':
                coalitions_mod = torch.from_numpy(coalitions)
                modality_masks[mod] = torch.cat((
                    coalitions_mod, 
                    torch.ones(n_coalitions, 1)), 
                    axis=1
                ).int().to(self.device)

            elif modality_type == 'image':
                coalitions_mod = torch.from_numpy(coalitions).to(self.device)
                if self.means:
                    modality_masks[mod] = self._expand_binary_mask(coalitions_mod, mod)
                else:
                    modality_masks[mod] = self._expand_binary_token_mask(coalitions_mod)

            batch_iters = n_coalitions // batch_size
            batch_left = n_coalitions % batch_size

            batches = []
            for batch_i in tqdm(range(batch_iters + 1)):            
                if batch_i < batch_iters:
                    batches.append(modality_masks[mod][(batch_i * batch_size):((batch_i + 1) * batch_size)])
                elif batch_left > 0:
                    batches.append(modality_masks[mod][(batch_i * batch_size):(batch_i * batch_size + batch_left)])
                else:
                    break

            modality_batches[mod] = {
                'iters': batch_iters,
                'left': batch_left,
                'batches': batches
            }

        repeat = 1
        tile = np.prod([len(b['batches']) for mod, b in modality_batches.items()])


        # print([(mod, b) for mod, b in modality_batches.items()])

        extended = {}
        for mod, b in modality_batches.items():
            batches = b['batches']
            tile = tile // len(batches)
            tiled = []
            for el in batches:
                tiled.extend([el] * tile)
            extended[mod] = (tiled * repeat)
            repeat *= len(batches)

        
        # print('extended', [(mod, len(b[0]), b[0]) for mod, b in extended.items()])

        outputs = np.zeros(tuple([*[modality_coalitions[mod].shape[0] for mod in self.modality_types], 1]), dtype=float)

        n_batches = np.prod(np.ceil(np.array(outputs.shape) / batch_size).astype(int))

        for i, idxs in tqdm(enumerate(np.ndindex(tuple(np.ceil(np.array(outputs.shape) / batch_size).astype(int)))), total=n_batches):
            mask = {self.mask_names[mod]: extended[mod][i] for mod in extended}
            out_prod = self.output_combinations(mask)
            slices = tuple(
                slice(idx * batch_size, (idx + 1) * batch_size)
                for idx in idxs
            )
            outputs[slices] = out_prod

        return outputs
    

    def output_combinations(self, modality_masks, batch_size=16):
        # Generate a matrix of outputs of each combination of inputs in the sen2l1c and coords modalities
        # Batch it using batch_size
        
        # 1. create a list of inputs (product of all image and text inputs)
        # 2. batch it and forward through the model using model.forward(inputs, mask=mask))
        # 3. concatenate and return

        combined_input = defaultdict(list)
        combined_masks = defaultdict(list)

        mask_sizes = [len(mask) for _, mask in modality_masks.items()]

        for idxs in np.ndindex(tuple(mask_sizes)):
            for i, mod in enumerate(modality_masks):
                combined_input[mod].append(self.inputs[self.mask_to_input_names[mod]])
                combined_masks[mod].append(modality_masks[mod][idxs[i]].unsqueeze(0))

        outputs = []

        for i in range(0, np.prod(mask_sizes), batch_size):
            if self.means:
                batch_inputs = {}

                for mod in modality_masks:
                    self.dev[mod] = {}
                    batch_mod = torch.concat(combined_input[mod][i:i+batch_size], axis=0)
                    mask_mod = torch.concat(combined_masks[mod][i:i+batch_size], axis=0)
                    b, c, h, w = batch_mod.shape

                    means = torch.tensor(self.means[self.mask_to_input_names[mod]]).to(self.device)
                    means = means.reshape(1, c, 1, 1).repeat((b, 1, h, w))
                    if self.means:
                        batch_inputs[self.mask_to_input_names[mod]] = batch_mod * mask_mod + means * mask_mod.logical_not()
                    else:
                        batch_inputs[self.mask_to_input_names[mod]] = batch_mod * mask_mod.logical_not() + means * mask_mod

                    self.dev[mod]['mask'] = mask_mod
                    self.dev[mod]['input'] = batch_inputs[self.mask_to_input_names[mod]]

                with torch.no_grad():
                    batch_outputs = self.model.forward(deepcopy(batch_inputs))
            else:
                batch_inputs = deepcopy({self.mask_to_input_names[mod]: torch.concat(combined_input[mod][i:i+batch_size], axis=0) for mod in modality_masks})
                batch_masks = deepcopy({mod: torch.concat(combined_masks[mod][i:i+batch_size], axis=0) for mod in modality_masks})

                with torch.no_grad():
                    batch_outputs = self.model.forward(batch_inputs, mask=batch_masks)

            self.dev['out'] = batch_outputs
            batch_outputs = self.aggregation_func(batch_outputs)

            outputs.append(batch_outputs)
        outputs = torch.concat(outputs, axis=0)
        return outputs.reshape(*mask_sizes, 1)
    

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


    def _expand_binary_token_mask(self, coalitions):
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
                .reshape(n_coalitions, self.true_grid_size * self.grid_step, self.true_grid_size * self.grid_step)[:, :self.grid_size, :self.grid_size]\
                .reshape(n_coalitions, -1)

        return binary_masks


    def _expand_binary_mask(self, coalitions, mod):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """

        _, c, h, w = self.inputs[mod].shape
        n_coalitions = coalitions.shape[0]
        n_players = int(np.sqrt(self.n_players_image[mod]))
        # Expand each coalition value into a patch
        binary_masks = coalitions\
            .repeat_interleave(self.grid_step**2, dim=1)\
                .reshape(n_coalitions, n_players, n_players, self.grid_step, self.grid_step)
        # Rearrange to form the final batch of full-size images
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, h, w)
        # Add image channel dimension
        binary_masks = binary_masks\
            .repeat((c, 1, 1, 1))\
                .permute(1, 0, 2, 3)
        return binary_masks
    

    def form_and_repeat_input(self, modality_masks):
        out = {}
        for mod, mask in modality_masks.items():
            n_coalitions = mask.shape[0]
            inp_mod_name = self.mask_to_input_names[mod]
            if self.modality_types[inp_mod_name] == 'text':
                out[inp_mod_name] = self.inputs[inp_mod_name].repeat(n_coalitions, 1)
            elif self.modality_types[inp_mod_name] == 'image':
                out[inp_mod_name] = self.inputs[inp_mod_name].repeat(n_coalitions, 1, 1, 1)

        return out