# Models -- CNNs, Llama-2-7B, Mixtral 8x7B, RNNT, DLRM, BERT-Large
import time
import random
import os
import pathlib
import torch
import torchaudio
from abc import ABC, abstractmethod
from PIL import Image
from datasets import load_dataset, Dataset
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from torchvision import transforms, models
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertForMaskedLM,
    BartForConditionalGeneration,
    ViTImageProcessor, 
    ViTModel,
    DetrImageProcessor,
    DetrForObjectDetection,
    AutoImageProcessor, 
    Mask2FormerForUniversalSegmentation,
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    AutoProcessor, 
    GPTJForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    RobertaForQuestionAnswering,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    MusicgenForConditionalGeneration,
    pipeline
)

class Inference(ABC):
    def __init__(self, model_name, device_id, batch_size):
        self._device = torch.device(f"cuda:{device_id}")
        self._model_name = model_name
        self._model = None
        self._batch_size = batch_size

    @abstractmethod
    def get_id(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def infer(self):
        pass

# Good to go
class CNN(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._imgs = None

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = torch.hub.load(
            "pytorch/vision:v0.14.1",
            self._model_name,
            verbose=False,
            pretrained=True
        )

        self._model.eval()
        self._model.to(self._device)

    def __load_numpy_data(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        input_tensor = preprocess(image)
        imgs = input_tensor.unsqueeze(0)
        self._imgs = imgs.repeat(self._batch_size, 1, 1, 1).pin_memory()

    def __load_non_numpy_data(self, image):
        # Orion profiler hack
        image = image.resize((256, 256))
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        right = (256 + 224) // 2
        bottom = (256 + 224) // 2
        image = image.crop((left, top, right, bottom))

        # Convert PIL image to tensor manually
        tensor_image = torch.tensor(
            [list(image.getdata())], dtype=torch.float32
        )
        # Reshape to CHW format
        tensor_image = tensor_image.view(1, 3, 224, 224)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor_image = (
            tensor_image - mean.view(1, 3, 1, 1)
        ) / std.view(1, 3, 1, 1)

        # Repeat the image batch_size times
        self._imgs = tensor_image.repeat(
            self._batch_size, 1, 1, 1
        ).pin_memory()

    def load_data(self):
        curr_path = pathlib.Path(__file__).parent.resolve()
        image_path = os.path.join(curr_path, "data/dog.jpg")
        image = Image.open(image_path)

        try:
            self.__load_numpy_data(image)
            self._to_infer = self._imgs.to(self._device, non_blocking=True)
        except RuntimeError:
            # Orion docker's torch is compiled without numpy support :(
            self.__load_non_numpy_data(image)
    
    def infer(self):
        with torch.no_grad():
            output = self._model(self._to_infer)
        torch.cuda.synchronize()
        return self._batch_size

# good to go
class StableDiffusion(Inference):
    def __init__(self, model_name, device_id, batch_size, height, width):
        super().__init__(model_name, device_id, batch_size)
        self._input_prompts = []
        self._prompts = [
            # "An astronaut riding a green horse",
            "Lebron James dunking in Mars",
            "Kobe Bryant versus Michael Jordan on the Moon",
            "MS Dhoni hitting a six to the Moon"
        ]
        self._model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        self._height = int(height)
        self._width = int(width)

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = DiffusionPipeline.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            use_safetensors=True, 
            variant="fp16"
        ).to(self._device)
    
    def load_data(self):
        # Prepare batch size number of prompts
        self._input_prompts = random.choices(self._prompts, k=self._batch_size)

    
    def infer(self):
        images = self._model(prompt=self._input_prompts, height=self._height, width=self._width).images
        torch.cuda.empty_cache()
        return len(images)

# Speech to Text
# Good to go
class Whisper(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._input_path = os.path.join(curr_path, "data/speech.wav")
        self._model_path = "openai/whisper-small"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = pipeline("automatic-speech-recognition", self._model_path, device=self._device)
    
    def load_data(self):
        # self._speeches = []
        # audio, sample_rate = torchaudio.load(self._input_path)
        # for _ in range(self._batch_size):
        #     # self._speeches.append({"raw": audio[0].numpy(), "sampling_rate": sample_rate})
        #     self._speeches.append(audio[0].numpy())
        
        audio_files = [self._input_path] * self._batch_size
        self._speeches = Dataset.from_dict({"audio": audio_files})

    def infer(self):
        transcriptions = self._model(inputs=self._speeches["audio"], batch_size=self._batch_size)
        return len(transcriptions)

# Text Generation
# Good to go
class GPT(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._input_prompts = []
        self._prompts = [
            "The NBA season is heating up with intense matchups and standout performances. Predictions for the NBA Finals?",
            "Discuss the impact of recent trades on the competitiveness of teams in the NBA Western Conference.",
            "Analyzing the rise of young talents in the NBA: Who are the top rookies to watch out for this season?",
            "Exploring the debate: Is LeBron James still the most dominant player in the NBA, or are there rising stars challenging his throne?"
        ]
        self.model_path = "EleutherAI/gpt-j-6B"
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    # def load_model(self):
    #     self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    #     self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     self._model = GPTJForCausalLM.from_pretrained(
    #         self.model_path,
    #         revision="float16",
    #         torch_dtype=torch.float16,
    #         ignore_mismatched_sizes=True # Supresses warnings about unused weights
    #     ).to(self._device)

    # def load_data(self):
    #     # Prepare batch size number of prompts
    #     self._input_prompts = random.choices(self._prompts, k=self._batch_size)
    #     self._tokenized_prompts = self._tokenizer(self._input_prompts, return_tensors="pt", padding=True, truncation=True)

    #     # Move inputs to GPU if available
    #     # self._tokenized_prompts.to(self._device)
    #     self._tokenized_prompts = {k: v.to(self._device) for k, v in self._tokenized_prompts.items()}

    # def infer(self):
    #     outputs = self._model.generate(
    #         **self._tokenized_prompts,
    #         max_length=256, # Following Power-Serve with their sequence length for LLMs
    #         num_return_sequences=len(self._input_prompts), 
    #         do_sample=True, 
    #         temperature=0.7,
    #         top_k=50,  
    #         top_p=0.95, 
    #         pad_token_id=self._tokenizer.eos_token_id  
    #     )
    #     # Decode generated responses
    #     generated_responses = [self._tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    #     # Print generated responses
    #     # for i, response in enumerate(generated_responses):
    #     #     print(f"Prompt {i+1}: {self._input_prompts[i]}")
    #     #     print(f"Generated Response: {response}\n")
    #     return len(generated_responses)
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self._model = GPTJForCausalLM.from_pretrained(
            self.model_path,
            revision="float16",
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True
        ).to(self._device)

        # Resize embeddings to include new pad token
        self._model.resize_token_embeddings(len(self._tokenizer))

    def load_data(self):
        self._input_prompts = random.choices(self._prompts, k=self._batch_size)
        self._tokenized_prompts = self._tokenizer(
            self._input_prompts,
            return_tensors="pt",
            padding="max_length",   # ensures equal sequence lengths
            truncation=True,
            max_length=128
        )
        self._tokenized_prompts = {k: v.to(self._device) for k, v in self._tokenized_prompts.items()}

    def infer(self):
        outputs = self._model.generate(
            **self._tokenized_prompts,
            max_length=256,
            num_return_sequences=self._batch_size,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self._tokenizer.pad_token_id,
            use_cache=True
        )
        responses = [self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        return len(responses)

# Object Detection - MLCommons
# Good to go
class RetinaNet(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._image_path = os.path.join(curr_path, "data/retinanet")
        self._confidence_threshold = 0.5
    
    def __load_numpy_data(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((800,800)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        input_tensor = preprocess(image)
        img = input_tensor.unsqueeze(0)
        return img
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        self._model.eval()
        self._model.to(self._device)

    def load_data(self):
        images = []
        image_path = [os.path.join(self._image_path, f) for f in os.listdir(self._image_path) if f.endswith('.jpg')]
        random.shuffle(image_path)
        for image in image_path[:self._batch_size]:
            img = Image.open(image).convert("RGB")
            img = self.__load_numpy_data(img)
            img = img.squeeze(0)
            img = img.to(self._device, non_blocking=True)
            images.append(img)
        
        # self._imgs = torch.stack(images).pin_memory()
        # self._to_infer = self._imgs.to(self._device, non_blocking=True)
        self._to_infer = images

    def infer(self):
        with torch.no_grad():
            output = self._model(self._to_infer)
        return len(output)

# Image Classification - https://huggingface.co/google/vit-base-patch16-224-in21k
# Needs testing
class Vit(Inference):
    def __init__(self, model_name, device_id, batch_size, resolution):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = 'google/vit-base-patch16-224-in21k'
        self._resolution = int(resolution)
        if self._resolution == 384:
            self._model_path = 'google/vit-large-patch32-384'
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._image_path = os.path.join(curr_path, "data/imagenet")
        # self._image_path = "./data/imagenet"

    def __load_numpy_data(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((self._resolution,self._resolution)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess(image)
        # img = input_tensor.unsqueeze(0)
        # return img
        return input_tensor
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = ViTModel.from_pretrained(self._model_path).to(self._device)
        self._processor = ViTImageProcessor.from_pretrained(self._model_path)

    def load_data(self):
        images = []
        image_path = [os.path.join(self._image_path, f) for f in os.listdir(self._image_path) if f.endswith('.JPEG')]
        random.shuffle(image_path)
        for image in image_path[:self._batch_size]:
            img = Image.open(image).convert("RGB")
            img = self.__load_numpy_data(img)
            images.append(img)
        
        self._imgs = torch.stack(images).pin_memory()
        self._to_infer = self._imgs.to(self._device, non_blocking=True)
    
    def infer(self):
        with torch.no_grad():
            self._model(self._to_infer)
        return self._batch_size

# Object Detection - https://huggingface.co/facebook/detr-resnet-50
# Good to go
class Detr(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = 'facebook/detr-resnet-50'
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._image_path = os.path.join(curr_path, "data/imagenet")
        # self._image_path = "./data/imagenet"
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = DetrForObjectDetection.from_pretrained(self._model_path).to(self._device)
        self._model.eval()
        self._processor = DetrImageProcessor.from_pretrained(self._model_path)

    def load_data(self):
        images = []
        image_path = [os.path.join(self._image_path, f) for f in os.listdir(self._image_path) if f.endswith('.JPEG')]
        random.shuffle(image_path)
        for image in image_path[:self._batch_size]:
            img = Image.open(image).convert("RGB")
            images.append(img)

        self._to_infer = self._processor(images=images, return_tensors="pt", padding=True).to(self._device)
    
    def infer(self):
        with torch.no_grad():
            outputs = self._model(**self._to_infer)
        return self._batch_size
    
# Image Segmentation
# Good to go
class Mask2Former(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = 'facebook/mask2former-swin-large-cityscapes-semantic'
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._image_path = os.path.join(curr_path, "data/coco")
        # self._image_path = './data/coco'

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(self._model_path).to(self._device)
        self._processor = AutoImageProcessor.from_pretrained(self._model_path)
        self._model.eval()

    def load_data(self):
        images = []
        image_path = [os.path.join(self._image_path, f) for f in os.listdir(self._image_path) if f.endswith('.jpg')]
        random.shuffle(image_path)
        for image in image_path[:self._batch_size]:
            img = Image.open(image).convert("RGB")
            images.append(img)

        self._to_infer = self._processor(images=images, return_tensors="pt", padding=True).to(self._device)
    
    def infer(self):
        with torch.no_grad():
            outputs = self._model(**self._to_infer)
        return outputs.class_queries_logits.shape[0]
    
# Image to Text - https://huggingface.co/microsoft/trocr-base-handwritten
# Good to go
class I2T(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = "microsoft/trocr-base-handwritten"
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._image_path = os.path.join(curr_path, "data/iam")
        # self._image_path = "./data/iam"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_path).to(self._device)
        self._model.eval()
        self._processor = TrOCRProcessor.from_pretrained(self._model_path)

    def load_data(self):
        images = []
        image_path = [os.path.join(self._image_path, f) for f in os.listdir(self._image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_path)
        for image in image_path[:self._batch_size]:
            img = Image.open(image).convert("RGB")
            images.append(img)

        self._to_infer = self._processor(images=images, return_tensors="pt").to(self._device)
    
    def infer(self):
        with torch.no_grad():
            outputs = self._model.generate(**self._to_infer, max_new_tokens=100)
        predictions = [self._processor.batch_decode(output, skip_special_tokens=True) for output in outputs]
        return len(predictions)

# Image to Image - https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
# Good to go
class SDXLRefiner(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
        curr_path = pathlib.Path(__file__).parent.resolve()
        image1 = os.path.join(curr_path, "data/sdxl/000000008.png")
        image2 = os.path.join(curr_path, "data/sdxl/000000002.png")
        self._inputs = [
            (image1, "a photo of an astronaut riding a green horse on mars"),
            (image2, "a photo of an astronaut riding a brown horse on mars")
        ]
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self._model_path, 
            ttorch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to(self._device)
    
    def load_data(self):
        prompts = []
        images = []

        self._final_inputs = random.choices(self._inputs, k=self._batch_size)
        for input in self._final_inputs:
            prompts.append(input[1])
            images.append(Image.open(input[0]).convert("RGB"))
        
        self._to_infer = {"prompt": prompts, "image": images}

    def infer(self):
        with torch.no_grad():
            outputs = self._model(**self._to_infer)
        return len(outputs.images)

# Text Classification -- BERT based (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
# Good to go
class BERTTextClassification(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = "cardiffnlp/twitter-roberta-base-sentiment"
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._text_path = os.path.join(curr_path, "data/text-classification/sentiment.txt")
        # self._text_path = "./data/text-classification/sentiment.txt"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path).to(self._device)
        self._model.eval()
    
    def load_data(self):
        with open(self._text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        random.shuffle(lines)
        selected_texts = lines[:self._batch_size]

        self._encoded_texts = self._tokenizer(selected_texts, padding=True, return_tensors="pt").to(self._device)

    def infer(self):
        with torch.no_grad():
            outputs = self._model(**self._encoded_texts)
            # predictions = torch.argmax(outputs.logits, dim=1) 
        # predictions = predictions.cpu().tolist()
        # print(predictions)
        # print(len(predictions))
        return self._batch_size

# Token classificaiton
# Good to go
class BertTokenClassification(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = "dslim/bert-base-NER"
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._text_path = os.path.join(curr_path, "data/token-classification/samples.txt")
        # self._text_path = "./data/token-classification/samples.txt"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = AutoModelForTokenClassification.from_pretrained(self._model_path).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model.eval()

    def load_data(self):
        with open(self._text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        random.shuffle(lines)
        selected_texts = lines[:self._batch_size]
        self._encoded_texts = self._tokenizer(selected_texts, return_tensors="pt", padding=True, truncation=True).to(self._device)

    def infer(self):
        with torch.no_grad():
            outputs = self._model(**self._encoded_texts)
        # predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        # return predictions
        return self._batch_size

# Question Answering - gotten from TieBreaker
# Good to go
class RobertaBatchedInference(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self.prompts = [
            "What is the capital of France?",
            "Who wrote 'Romeo and Juliet'?",
            "What is the tallest mountain in the world?",
            "Who discovered electricity?",
            "What is the chemical formula for water?",
            "Who painted the Mona Lisa?",
            "What is the currency of Japan?",
            "Who is the current president of the United States?",
            "What is the largest planet in our solar system?",
            "Who invented the telephone?",
            "What is the currency of Australia?",
            "Who was the first man to walk on the moon?",
            "What is the speed of light?",
            "Who is the author of 'To Kill a Mockingbird'?",
            "What is the main ingredient in sushi?",
            "Who composed the 'Moonlight Sonata'?",
            "What is the population of China?",
            "Who was the first woman to win a Nobel Prize?",
            "What is the atomic number of carbon?",
            "Who is the CEO of Tesla?",
            "What is the national animal of Australia?",
            "Who was the first President of the United States?",
            "What is the largest ocean on Earth?",
            "Who discovered penicillin?",
            "What is the boiling point of water?",
            "Who painted 'Starry Night'?",
            "What is the capital of Russia?",
            "Who wrote '1984'?",
            "What is the longest river in the world?",
            "Who developed the theory of relativity?",
            "What is the tallest building in the world?",
            "Who is the current Prime Minister of the United Kingdom?",
            "What is the chemical symbol for gold?",
            "Who founded Microsoft?",
            "What is the temperature at which Fahrenheit and Celsius scales are equal?",
            "Who is the Greek god of the sea?",
            "What is the capital of Brazil?",
            "Who directed the movie 'The Shawshank Redemption'?",
            "What is the largest desert in the world?",
            "Who was the first female Prime Minister of the United Kingdom?"
        ]
        self.contexts = [
            "The capital city of France is Paris.",
            "'Romeo and Juliet' was written by William Shakespeare.",
            "Mount Everest is the tallest mountain in the world.",
            "Electricity was discovered by Benjamin Franklin.",
            "The chemical formula for water is H2O.",
            "The Mona Lisa was painted by Leonardo da Vinci.",
            "The currency of Japan is the Japanese yen.",
            "The current president of the United States is Joe Biden.",
            "Jupiter is the largest planet in our solar system.",
            "The telephone was invented by Alexander Graham Bell.",
            "The currency of Australia is the Australian dollar.",
            "The first man to walk on the moon was Neil Armstrong.",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
            "Harper Lee is the author of 'To Kill a Mockingbird'.",
            "The main ingredient in sushi is rice.",
            "Ludwig van Beethoven composed the 'Moonlight Sonata'.",
            "The population of China is over 1.4 billion people.",
            "Marie Curie was the first woman to win a Nobel Prize.",
            "The atomic number of carbon is 6.",
            "Elon Musk is the CEO of Tesla.",
            "The national animal of Australia is the kangaroo.",
            "George Washington was the first President of the United States.",
            "The largest ocean on Earth is the Pacific Ocean.",
            "Alexander Fleming discovered penicillin.",
            "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.",
            "Vincent van Gogh painted 'Starry Night'.",
            "The capital of Russia is Moscow.",
            "George Orwell wrote '1984'.",
            "The longest river in the world is the Nile River.",
            "Albert Einstein developed the theory of relativity.",
            "The tallest building in the world is the Burj Khalifa in Dubai, United Arab Emirates.",
            "The current Prime Minister of the United Kingdom is Boris Johnson.",
            "The chemical symbol for gold is Au.",
            "Microsoft was founded by Bill Gates and Paul Allen.",
            "The temperature at which Fahrenheit and Celsius scales are equal is -40 degrees.",
            "Poseidon is the Greek god of the sea.",
            "The capital of Brazil is Brasília.",
            "Frank Darabont directed the movie 'The Shawshank Redemption'.",
            "The largest desert in the world is the Sahara Desert.",
            "Margaret Thatcher was the first female Prime Minister of the United Kingdom."
        ]
        self.model_path = "deepset/roberta-base-squad2"
        self.tokenizer = None
        self.inputs = None
        self.selected_prompts = None
        self.selected_texts = None

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = RobertaForQuestionAnswering.from_pretrained(
            self.model_path
        ).to(self._device)

    def load_data(self):
        self.selected_prompts = random.sample(self.prompts, self._batch_size)
        self.selected_texts = [self.contexts[self.prompts.index(prompt)] for prompt in self.selected_prompts]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.inputs = self.tokenizer(self.selected_prompts, self.selected_texts, padding=True, truncation=True, return_tensors="pt")

    def infer(self):
        self.inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model(**self.inputs)

        # # Process outputs for each question-text pair
        # total_tokens_generated = 0
        # for i, (prompt, text) in enumerate(zip(self.selected_prompts, self.selected_texts)):
        #     answer_start_index = outputs.start_logits[i].argmax()
        #     answer_end_index = outputs.end_logits[i].argmax()
        #     predict_answer_tokens = self.inputs.input_ids[i, answer_start_index:answer_end_index + 1]
        #     num_tokens_generated = len(predict_answer_tokens)
        #     total_tokens_generated += num_tokens_generated
        # return total_tokens_generated
        # torch.cuda.synchronize()
        return self._batch_size

# Zero-shot classification

# Translation - https://chatgpt.com/c/67e0cd9c-8c4c-8011-b8dc-c06427926af3
# Good to go
class T5Model(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = 'google-t5/t5-base'
        self._tokenizer_path = 'google-t5/t5-base'
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._text_path = os.path.join(curr_path, "data/translation/samples.txt")
        # self._text_path = "./data/translation/samples.txt"
    
    def __tokenize_text(self, text):
        return self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self._max_length)
    
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = T5ForConditionalGeneration.from_pretrained(self._model_path).to(self._device)
        self._tokenizer = T5Tokenizer.from_pretrained(self._tokenizer_path)
    
    def load_data(self):
        # text_files = [os.path.join(self._text_path, f) for f in os.listdir(self._text_path) if f.endswith(".txt")]
        # random.shuffle(text_files)
        
        # texts = []
        # for file in text_files[:self._batch_size]:
        #     with open(file, "r", encoding="utf-8") as f:
        #         texts.append(f.read().strip())
        # tokenized_inputs = [self.__tokenize_text(text) for text in texts]
        # self._to_infer = {key: torch.cat([t[key] for t in tokenized_inputs]).to(self._device, non_blocking=True) for key in tokenized_inputs[0]}


        with open(self._text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        random.shuffle(lines)
        selected_texts = lines[:self._batch_size]
        self._to_infer = self._tokenizer(selected_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self._device)


    
    def infer(self):
        with torch.no_grad():
            outputs = self._model.generate(**self._to_infer, max_length=512)
        return self._batch_size

# Summarization
# Good to go
class TextSummarizer(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = 'facebook/bart-large-cnn'
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._text_path = os.path.join(curr_path, "data/summarization")
        # self._text_path = './data/summarization'

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_model(self):
        self._model = BartForConditionalGeneration.from_pretrained(self._model_path).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model.eval()

    def load_data(self):
        texts = []
        text_files = [os.path.join(self._text_path, f) for f in os.listdir(self._text_path) if f.endswith('.txt')]
        random.shuffle(text_files)
        
        for file in text_files[:self._batch_size]:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
        
        self._to_infer = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self._device)
    
    def infer(self):
        with torch.no_grad():
            outputs = self._model.generate(**self._to_infer, max_length=150, num_beams=4, early_stopping=True)
        return len(outputs)

# Text to Speech
# Good to go
class Text2Speech(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = "microsoft/speecht5_tts"
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._input_path = os.path.join(curr_path, "data/text2speech/samples-80-characters.txt")
        # self._input_path = "./data/text2speech/samples-80-characters.txt"

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_data(self):
        with open(self._input_path, "r") as file:
            prompts = file.readlines()
        prompts = [line.strip() for line in prompts]
        prompts = random.sample(prompts, self._batch_size)
        self._to_infer = self._processor(text=prompts, return_tensors="pt", padding=True).to(self._device)

    def load_model(self):
        self._model = SpeechT5ForTextToSpeech.from_pretrained(self._model_path).to(self._device)
        self._processor = SpeechT5Processor.from_pretrained(self._model_path)
        self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self._device)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self._speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self._device)
        # self._speaker_embeddings, _ = torchaudio.load(
        #     torchaudio.utils.download_asset("speechbrain/spkrec-xvect-voxceleb", "spkrec-xvect-voxceleb/1234.wav")
        # )
        # self._speaker_embeddings = self._speaker_embeddings.mean(dim=1).unsqueeze(0).to(self._device)
    
    def infer(self):
        with torch.no_grad():
            speech_outputs = self._model.generate(**self._to_infer, speaker_embeddings=self._speaker_embeddings, vcoder=self._vocoder)
        return len(speech_outputs)
        
# Text to Audio
# Good to go
class Text2Audio(Inference):
    def __init__(self, model_name, device_id, batch_size):
        super().__init__(model_name, device_id, batch_size)
        self._model_path = "facebook/musicgen-medium"
        curr_path = pathlib.Path(__file__).parent.resolve()
        self._input_path = os.path.join(curr_path, "data/text2music/prompts.txt")
        # self._input_path = "./data/text2music/prompts.txt"
        self._duration = 10

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"
    
    def load_data(self):
        with open(self._input_path, "r") as file:
            prompts = file.readlines()
        prompts = [line.strip() for line in prompts]
        prompts = random.sample(prompts, self._batch_size)
        self._to_infer = self._processor(text=prompts, return_tensors="pt", padding=True).to(self._device)

    def load_model(self):
        self._model = MusicgenForConditionalGeneration.from_pretrained(self._model_path).to(self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_path)
    
    def infer(self):
        with torch.no_grad():
            music_outputs = self._model.generate(**self._to_infer, max_new_tokens=int(self._duration * 50))
        return len(music_outputs)


def get_inference_object(model, device_id, batch_size):
    if "diffusion" in model:
        model_name = model.split("_")[0]
        height = model.split("_")[1]
        width = model.split("_")[2]
        return StableDiffusion(model_name, device_id, batch_size, height, width)
    elif model == "whisper":
        return Whisper(model, device_id, batch_size)
    elif model == "gpt":
        return GPT(model, device_id, batch_size)
    elif model == "retinanet":
        return RetinaNet(model, device_id, batch_size)
    elif "vit" in model:
        model_name = model.split("_")[0]
        resolution = model.split("_")[1]
        return Vit(model_name, device_id, batch_size, resolution)
    elif model == "detr":
        return Detr(model, device_id, batch_size)
    elif model == "mask2former":
        return Mask2Former(model, device_id, batch_size)
    elif model == "trocr":
        return I2T(model, device_id, batch_size)
    elif model == "sdxlrefine":
        return SDXLRefiner(model, device_id, batch_size)
    elif model == "berttext":
        return BERTTextClassification(model, device_id, batch_size)
    elif model == "berttoken":
        return BertTokenClassification(model, device_id, batch_size)
    elif model == "roberta":
        return RobertaBatchedInference(model, device_id, batch_size)
    elif model == "t5":
        return T5Model(model, device_id, batch_size)
    elif model == "bart":
        return TextSummarizer(model, device_id, batch_size)
    elif model == "speecht5":
        return Text2Speech(model, device_id, batch_size)
    elif model == "musicgen":
        return Text2Audio(model, device_id, batch_size)
    elif model == "mobilenet_v3_large" \
        or model == "resnet50" \
        or model == "efficientnet_b7" \
        or model == "resnet18" \
        or model == "vgg13":
        return CNN(model, device_id, batch_size)
    else:
        return CNN(model, device_id, batch_size)
    

# sd = StableDiffusion(
#     model_name="stable-diffusion-xl",
#     device_id=0,
#     batch_size=32,
#     height=256,
#     width=256
# )

# sd.load_model()

# for i in range(5):
#     sd.load_data()
#     count = sd.infer()
#     print(f"Inference {i+1}: generated {count} image(s)")
# # Create an instance of the class
# vision = Text2Audio(device_id=0, model_name="detr", batch_size=1)

# # Load the model and data
# vision.load_model()
# print('loaded model!')
# time.sleep(5)
# vision.load_data()
# print('loaded data!')
# time.sleep(5)

# # Run 10 inferences
# start = time.time()
# for _ in range(4):
#     # print('bout to infer!')
#     batch_size_returned = vision.infer()
#     # print('finished infering!')
#     # print(f"Batch size returned: {batch_size_returned}")
# end = time.time()
# throughput = 4 / (end - start)
# print(f"Throughput: {throughput}")
