import torch
import torch.nn as nn

from .encoder import Encoder
from .vq import VQ
from .decoder import ImgDecoder
from .decoder import SegDecoder
from .discriminator import LatentDiscriminator
from .discriminator import LatentDiscriminatorV2
from .discriminator import ImageDiscriminator
from .normal_classifier import NormalityClassifier
from .initialize import init_weights
from .utils import apply_spectral_norm


def init_models(input_dim: int = 4,
                img_output_dim: int = 4,
                seg_output_dim: int = 3,
                img_output_act: str = 'none',
                emb_dim: int = 64,
                dict_size: int = 512,
                enc_filters: list = [32, 64, 128, 128, 128, 128],
                dec_filters: list = [128, 128, 128, 128, 64, 32],
                latent_size: int = 8,
                init_type: str = 'kaiming',
                init_latent_discriminator: bool = False,
                latent_discriminator_type: str = 'V1',
                init_image_discriminator: bool = False,
                img_dis_filters: list = None,
                faiss_backend: str = 'torch',
                apply_spectral_norm_to_discriminator: bool = True,
                ):

    encoder = Encoder(input_dim=input_dim,
                      emb_dim=emb_dim,
                      filters=enc_filters)

    vq1 = VQ(emb_dim=emb_dim,
             dict_size=dict_size,
             momentum=0.99,
             eps=1e-5,
             knn_backend=faiss_backend)

    vq2 = VQ(emb_dim=emb_dim,
             dict_size=dict_size,
             momentum=0.99,
             eps=1e-5,
             knn_backend=faiss_backend)

    seg_decoder = SegDecoder(output_dim=seg_output_dim,
                             emb_dim=emb_dim,
                             filters=dec_filters)

    img_decoder = ImgDecoder(output_dim=img_output_dim,
                             img_output_act=img_output_act,
                             emb_dim=emb_dim,
                             label_channels=seg_output_dim,
                             filters=dec_filters)

    init_weights(encoder, init_type)
    init_weights(seg_decoder, init_type)
    init_weights(img_decoder, init_type)

    encoder.cuda()
    encoder = nn.DataParallel(encoder)

    vq1.cuda()
    vq1 = nn.DataParallel(vq1)

    vq2.cuda()
    vq2 = nn.DataParallel(vq2)

    seg_decoder.cuda()
    seg_decoder = nn.DataParallel(seg_decoder)

    img_decoder.cuda()
    img_decoder = nn.DataParallel(img_decoder)

    lat_dis = None
    if init_latent_discriminator:

        if latent_discriminator_type == 'V1':
            lat_dis = LatentDiscriminator(emb_dim=emb_dim,
                                          latent_size=latent_size)

        elif latent_discriminator_type == 'V2':
            lat_dis = LatentDiscriminatorV2(emb_dim=emb_dim,
                                            latent_size=latent_size)

        else:
            raise Exception(
                'Please specify latent discriminator type correctly.')

        print(lat_dis)

        if apply_spectral_norm_to_discriminator:
            apply_spectral_norm(lat_dis)

        init_weights(lat_dis, 'normal')

        lat_dis.cuda()
        lat_dis = nn.DataParallel(lat_dis)

    img_dis_1 = None
    img_dis_2 = None
    if init_image_discriminator:
        img_dis_1 = ImageDiscriminator(input_dim=input_dim,
                                       filters=img_dis_filters)
        img_dis_2 = ImageDiscriminator(input_dim=input_dim,
                                       filters=img_dis_filters)

        if apply_spectral_norm_to_discriminator:
            apply_spectral_norm(img_dis_1)
            apply_spectral_norm(img_dis_2)

        init_weights(img_dis_1, 'normal')
        init_weights(img_dis_2, 'normal')

        img_dis_1.cuda()
        img_dis_1 = nn.DataParallel(img_dis_1)

        img_dis_2.cuda()
        img_dis_2 = nn.DataParallel(img_dis_2)

    return encoder, vq1, vq2, seg_decoder, img_decoder, lat_dis, img_dis_1, img_dis_2
# import torch
# import torch.nn as nn

# from .encoder import Encoder
# from .vq import VQ
# from .decoder import ImgDecoder
# from .decoder import SegDecoder
# from .discriminator import LatentDiscriminator
# from .discriminator import LatentDiscriminatorV2
# from .discriminator import ImageDiscriminator
# from .normal_classifier import NormalityClassifier
# from .initialize import init_weights
# from .utils import apply_spectral_norm

# def setup_device():
#     """Determine if a GPU is available and return the appropriate device object."""
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def init_models(input_dim: int = 4,
#                 img_output_dim: int = 4,
#                 seg_output_dim: int = 3,
#                 img_output_act: str = 'none',
#                 emb_dim: int = 64,
#                 dict_size: int = 512,
#                 enc_filters: list = [32, 64, 128, 128, 128, 128],
#                 dec_filters: list = [128, 128, 128, 128, 64, 32],
#                 latent_size: int = 8,
#                 init_type: str = 'kaiming',
#                 init_latent_discriminator: bool = False,
#                 latent_discriminator_type: str = 'V1',
#                 init_image_discriminator: bool = False,
#                 img_dis_filters: list = None,
#                 faiss_backend: str = 'torch',
#                 apply_spectral_norm_to_discriminator: bool = True):
#     device = setup_device()
#     print(f"Using device: {device}")

#     encoder = Encoder(input_dim=input_dim, emb_dim=emb_dim, filters=enc_filters).to(device)
#     vq1 = VQ(emb_dim=emb_dim, dict_size=dict_size, momentum=0.99, eps=1e-5, knn_backend=faiss_backend).to(device)
#     vq2 = VQ(emb_dim=emb_dim, dict_size=dict_size, momentum=0.99, eps=1e-5, knn_backend=faiss_backend).to(device)
#     seg_decoder = SegDecoder(output_dim=seg_output_dim, emb_dim=emb_dim, filters=dec_filters).to(device)
#     img_decoder = ImgDecoder(output_dim=img_output_dim, img_output_act=img_output_act, emb_dim=emb_dim, filters=dec_filters).to(device)

#     init_weights(encoder, init_type)
#     init_weights(seg_decoder, init_type)
#     init_weights(img_decoder, init_type)

#     # Setting up the latent and image discriminators based on the configuration.
#     lat_dis = setup_discriminator(init_latent_discriminator, latent_discriminator_type, emb_dim, latent_size, device)
#     img_dis_1, img_dis_2 = setup_image_discriminators(init_image_discriminator, input_dim, img_dis_filters, device)

#     return encoder, vq1, vq2, seg_decoder, img_decoder, lat_dis, img_dis_1, img_dis_2

# def setup_discriminator(init, discriminator_type, emb_dim, latent_size, device):
#     if init:
#         if discriminator_type == 'V1':
#             discriminator = LatentDiscriminator(emb_dim=emb_dim, latent_size=latent_size).to(device)
#         elif discriminator_type == 'V2':
#             discriminator = LatentDiscriminatorV2(emb_dim=emb_dim, latent_size=latent_size).to(device)
#         else:
#             raise Exception('Please specify latent discriminator type correctly.')

#         if apply_spectral_norm_to_discriminator:
#             apply_spectral_norm(discriminator)

#         init_weights(discriminator, 'normal')
#         return discriminator

# def setup_image_discriminators(init, input_dim, filters, device):
#     if init:
#         img_dis_1 = ImageDiscriminator(input_dim=input_dim, filters=filters).to(device)
#         img_dis_2 = ImageDiscriminator(input_dim=input_dim, filters=filters).to(device)

#         if apply_spectral_norm_to_discriminator:
#             apply_spectral_norm(img_dis_1)
#             apply_spectral_norm(img_dis_2)

#         init_weights(img_dis_1, 'normal')
#         init_weights(img_dis_2, 'normal')
        
#         return img_dis_1, img_dis_2
#     return None, None

# # Complete the setup of your initialization models function using the defined helper functions
# def init_models(input_dim: int = 4,
#                 img_output_dim: int = 4,
#                 seg_output_dim: int = 3,
#                 img_output_act: str = 'none',
#                 emb_dim: int = 64,
#                 dict_size: int = 512,
#                 enc_filters: list = [32, 64, 128, 128, 128, 128],
#                 dec_filters: list = [128, 128, 128, 128, 64, 32],
#                 latent_size: int = 8,
#                 init_type: str = 'kaiming',
#                 init_latent_discriminator: bool = False,
#                 latent_discriminator_type: str = 'V1',
#                 init_image_discriminator: bool = False,
#                 img_dis_filters: list = None,
#                 faiss_backend: str = 'torch',
#                 apply_spectral_norm_to_discriminator: bool = True):
#     device = setup_device()
#     print(f"Using device: {device}")

#     encoder = Encoder(input_dim=input_dim, emb_dim=emb_dim, filters=enc_filters).to(device)
#     vq1 = VQ(emb_dim=emb_dim, dict_size=dict_size, momentum=0.99, eps=1e-5, knn_backend=faiss_backend).to(device)
#     vq2 = VQ(emb_dim=emb_dim, dict_size=dict_size, momentum=0.99, eps=1e-5, knn_backend=faiss_backend).to(device)
#     seg_decoder = SegDecoder(output_dim=seg_output_dim, emb_dim=emb_dim, filters=dec_filters).to(device)
#     img_decoder = ImgDecoder(output_dim=img_output_dim, img_output_act=img_output_act, emb_dim=emb_dim, filters=dec_filters).to(device)

#     init_weights(encoder, init_type)
#     init_weights(seg_decoder, init_type)
#     init_weights(img_decoder, init_type)

#     lat_dis = setup_discriminator(init_latent_discriminator, latent_discriminator_type, emb_dim, latent_size, device)
#     img_dis_1, img_dis_2 = setup_image_discriminators(init_image_discriminator, input_dim, img_dis_filters, device)

#     return encoder, vq1, vq2, seg_decoder, img_decoder, lat_dis, img_dis_1, img_dis_2

