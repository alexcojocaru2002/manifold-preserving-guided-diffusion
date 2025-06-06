import clip
import librosa
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wav2clip


class AudioImageLoss(torch.nn.Module):
    def __init__(self, audio_path, device="cuda"):
        super().__init__()
        self.device = device

        # CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_preprocessing = self._get_clip_preprocessing()

        # wav2CLIP
        # self.wav2clip = wav2clip.get_model(device=self.device)
        self.wav2clip = wav2clip.get_model()
        self.wav2clip.eval()
        for p in self.wav2clip.parameters():
            p.requires_grad = False

        self.audio_embedding = self._get_audio_embedding(audio_path)

    def _get_clip_preprocessing(self):
        return T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),  # resize shorter side to 224
                T.CenterCrop(224),  # make it exactly 224×224
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def _get_audio_embedding(self, audio_path):
        wav, sr = librosa.load(audio_path, sr=48000, mono=True)
        with torch.no_grad():
            audio_embedding = wav2clip.embed_audio(wav, self.wav2clip)

        # embedding is batched (1, 512)
        return audio_embedding

    def _get_image_embedding(self, image: torch.Tensor):
        preprocessed_image = self.clip_preprocessing(image)
        image_embedding = self.clip_model.encode_image(preprocessed_image)
        
        # embedding is batched (1, 512)
        return image_embedding

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image_embedding = self._get_image_embedding(image)
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)(image_embedding, self.audio_embedding)
        loss = 1.0 - cosine_similarity
        
        # Taking the mean just to unsqueeze batch dimension
        return loss.mean()

