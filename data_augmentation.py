import random
from contextlib import contextmanager

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine

# Mild or strong augmentation select in main.py (augmentation = ...)
class RandomAffinePair:
    def __init__(
        self,
        degrees: float = 5.0,
        translate_fraction: float = 0.03,
        scale_range: tuple[float, float] = (0.95, 1.05),
    ):
        if degrees < 0:
            raise ValueError("degrees must be non-negative")
        if not 0 <= translate_fraction <= 1:
            raise ValueError("translate_fraction must be between 0 and 1")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError("scale_range must contain positive, ordered values")

        self.degrees = degrees
        self.translate_fraction = translate_fraction
        self.scale_range = scale_range
        self._last_parameters = None
        self._sample_active = False

    def _sample_parameters(self, image: Image.Image):
        width, height = image.size
        angle = random.uniform(-self.degrees, self.degrees)
        translate = (
            int(round(random.uniform(-self.translate_fraction, self.translate_fraction) * width)),
            int(round(random.uniform(-self.translate_fraction, self.translate_fraction) * height)),
        )
        scale = random.uniform(*self.scale_range)
        return angle, translate, scale

    @staticmethod
    def _apply(image, parameters, interpolation):
        angle, translate, scale = parameters
        return affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=interpolation,
            fill=0,
        )

    @contextmanager
    def sample(self):
        if self._sample_active:
            raise RuntimeError("Augmentation samples cannot be nested")
        self._sample_active = True
        self._last_parameters = None
        try:
            yield
        finally:
            self._sample_active = False
            self._last_parameters = None

    def transform_image(self, image: Image.Image) -> Image.Image:
        if not self._sample_active or self._last_parameters is None:
            self._last_parameters = self._sample_parameters(image)
        return self._apply(
            image, self._last_parameters, InterpolationMode.BILINEAR
        )

    def transform_mask(self, mask: Image.Image) -> Image.Image:
        if self._last_parameters is None:
            raise RuntimeError("transform_image must be called before transform_mask")
        return self._apply(
            mask, self._last_parameters, InterpolationMode.NEAREST
        )

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if image.size != mask.size:
            raise ValueError(
                f"Image and mask sizes must match, got {image.size} and {mask.size}"
            )

        parameters = self._sample_parameters(image)
        return (
            self._apply(image, parameters, InterpolationMode.BILINEAR),
            self._apply(mask, parameters, InterpolationMode.NEAREST),
        )


def make_augmentation(profile):
    if profile == 'none':
        return None
    if profile == 'mild':
        return RandomAffinePair()
    if profile == 'strong':
        return RandomAffinePair(degrees=10, translate_fraction=0.05,
                                scale_range=(0.9, 1.1))
    raise ValueError(f'Unknown augmentation profile: {profile}')
