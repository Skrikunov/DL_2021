# Feel free to add functions, classes...

import lj_speech


def train_tts(dataset_root, num_epochs):
    """
    Train the TTS system from scratch on LJ-Speech-aligned stored at
    `dataset_root` for `num_epochs` epochs and save the best model to
    (!!! 'best' in terms of audio quality!) "./TTS.pth".

    dataset_root:
        `pathlib.Path`
        The argument for `lj_speech.get_dataset()`.
    """
    ## ...
    ## ...


class TextToSpeechSynthesizer:
    """
    Inference-only interface to the TTS model.
    """
    def __init__(self, checkpoint_path):
        """
        Create the TTS model on GPU, loading its weights from `checkpoint_path`.

        checkpoint_path:
            `str`
        """
        self.vocoder = lj_speech.Vocoder()
        ## ...
        ## ...

    def synthesize_from_text(self, text):
        """
        Synthesize text into voice.

        text:
            `str`

        return:
        audio:
            `torch.Tensor` or `numpy.ndarray`, shape == (1, t)
        """
        phonemes = lj_speech.text_to_phonemes(text)
        return self.synthesize_from_phonemes(phonemes)

    def synthesize_from_phonemes(self, phonemes, durations=None):
        """
        Synthesize phonemes into voice.

        phonemes:
            `list` of `str`
            ARPAbet phoneme codes.
        durations:
            `list` of `int`, optional
            Duration in spectrogram frames for each phoneme.
            If given, used for alignment in the model (like during
            training); otherwise, durations are predicted by the duration
            model.

        return:
        audio:
            torch.Tensor or numpy.ndarray, shape == (1, t)
        """
        ## ...
        ## ...

        spectrogram = ## ...

        return self.vocoder(spectrogram)


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'TTS.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'TTS.pth'.
        On Linux (in Colab too), use `$ md5sum TTS.pth`.
        On Windows, use `> CertUtil -hashfile TTS.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'TTS.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here
    # Example: `md5_checksum = "747822ca4436819145de8f9e410ca9ca"`
    # Example: `google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"

    return md5_checksum, google_drive_link
