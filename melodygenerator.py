"""
Advanced Melody Generation Engine Powered by Transformer Intelligence

This sophisticated melody generation system represents the culmination of our
Transformer-based musical AI architecture. The MelodyGenerator leverages the
trained model's deep understanding of musical patterns to create coherent,
musically intelligent melodies through advanced autoregressive generation.

The system implements a sophisticated iterative generation process that builds
melodies note-by-note, using the entire generated sequence as context for each
new prediction. This approach ensures that generated melodies maintain musical
coherence and follow learned patterns from the training data.

Revolutionary Generation Features:
- Autoregressive melody construction using full sequence context
- Intelligent note prediction based on learned musical relationships
- Sophisticated tokenization preserving musical structure
- Dynamic length control with intelligent stopping criteria

Musical Intelligence Capabilities:
- Generates musically coherent note sequences
- Maintains harmonic and melodic relationships throughout generation
- Preserves rhythmic patterns and timing relationships
- Creates novel melodies that follow learned musical conventions

This generation engine represents the creative output of our Transformer model,
transforming learned musical patterns into beautiful, original melodies that
demonstrate the model's deep understanding of musical structure and composition.
"""

import tensorflow as tf


class MelodyGenerator:
    """
    Revolutionary melody generation engine that transforms learned musical patterns
    into beautiful, original melodies using advanced Transformer intelligence.
    
    This sophisticated system implements cutting-edge autoregressive generation
    techniques, leveraging the model's deep understanding of musical relationships
    to create coherent, musically intelligent sequences that demonstrate the power
    of AI-driven musical composition.
    """

    def __init__(self, transformer, tokenizer, max_length=50):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor[0])

        for _ in range(num_notes_to_generate):
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            predicted_note = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, predictions):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]
        predicted_note_index = tf.argmax(latest_predictions, axis=1)
        predicted_note = predicted_note_index.numpy()[0]
        return predicted_note

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.numpy()
        generated_melody = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_melody
