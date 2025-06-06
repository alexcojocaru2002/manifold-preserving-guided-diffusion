import torch
import torch.nn.functional as F
import torchvision.transforms as T
import librosa
import wav2clip
import clip


class AudioImageLoss(torch.nn.Module):
    def __init__(self, audio_path, device="cuda"):
        super().__init__()
        self.device = device

        # CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        # freeze CLIP
        # for p in self.clip_model.parameters():
        #     p.requires_grad = False
        self.clip_preprocess = self._get_clip_preprocess()

        # wav2CLIP
        self.wav2clip_model = wav2clip.get_model().to(device)

        self.audio_embedding = self._get_embedding_from_audio(audio_path)

    def _get_clip_preprocess(self):
        return T.Compose(
            [
                T.Resize(224),  # resize shorter side to 224
                T.CenterCrop(224),  # make it exactly 224×224
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def preprocess_tensor_for_clip(self, img_tensor):
        # img_tensor: (1, 3, H, W), values in [0, 1], torch.Tensor
        img_tensor = F.interpolate(
            img_tensor, size=224, mode="bilinear", align_corners=False
        )
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=img_tensor.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=img_tensor.device
        ).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        return img_tensor

    def _get_embedding_from_audio(self, audio_path):
        wav, sr = librosa.load(audio_path, sr=48000, mono=True)
        # with torch.no_grad():
        audio_embedding_numpy = wav2clip.embed_audio(
            wav, self.wav2clip_model
        )  # np.ndarray (1,512)
        audio_embedding = torch.from_numpy(audio_embedding_numpy).to(self.device)
        audio_embedding = F.normalize(audio_embedding, dim=-1)

        return audio_embedding

    def __call__(self, clean_image_estimation: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: torch.Tensor, shape (1, 3, H, W), normalized for CLIP
        audio_path:   path to wav file (mono)
        returns:      scalar loss = 1 - cosine(u_i, u_a)
        """
        # with torch.no_grad():
        image_embedding = self.clip_model.encode_image(
            self.preprocess_tensor_for_clip(clean_image_estimation)
        )  # (1,512)
        image_embedding = F.normalize(image_embedding, dim=-1)
        # image_embedding = image_embedding.unsqueeze(0)

        # 3) loss = 1 - cosine similarity
        cos = torch.sum(image_embedding * self.audio_embedding, dim=-1)  # (1,)
        loss = 1.0 - cos
        return loss.mean()
