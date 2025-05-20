import random
import time
from typing import List, Tuple, Dict, Set
import numpy as np
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
import os

from dataset.dataset_reader import read_dataset
from models.model_abstraction import Model

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class IBMModel1(Model):
    """
    IBM Model 1 for statistical machine translation.
    
    This implementation follows the EM algorithm for learning word alignment
    probabilities between source and target languages.
    """
    
    # Class constants
    TRAINING_DATA_PERCENTAGE = 0.8
    SENTENCE_LIMIT = 100_000
    NUM_ITERATIONS = 10
    EPSILON = 1e-12
    CHECKPOINT_FILE = "ibm_model1_checkpoint.npz"
    
    def __init__(self, seed: int = 42):
        """
        Initialize the IBM Model 1.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Load and prepare dataset
        self.dataset = read_dataset()
        random.shuffle(self.dataset)
        
        # Initialize vocabularies and mappings
        self.italian_vocab: Set[str] = set()
        self.english_vocab: Set[str] = set()
        self.italian_mapping: Dict[str, int] = {}
        self.english_mapping: Dict[str, int] = {}
        
        # Build vocabularies from the dataset
        self._build_vocabularies()
        
        print(f"Italian unique words: {len(self.italian_vocab)}")
        print(f"English unique words: {len(self.english_vocab)}")
        
        # Initialize translation probabilities uniformly
        self.translation_probabilities = self._initialize_translation_probabilities()
        self.current_iteration = 0
    
    def _build_vocabularies(self) -> None:
        """Build vocabularies and word-to-index mappings from the dataset."""
        italian_counter = 1
        english_counter = 1
        
        # Process limited number of sentences for vocabulary building
        for english_sentence, italian_sentence in self.dataset[:self.SENTENCE_LIMIT]:
            # Process English words
            for word in english_sentence.split():
                if word not in self.english_vocab:
                    self.english_mapping[word] = english_counter
                    english_counter += 1
                    self.english_vocab.add(word)
            
            # Process Italian words
            for word in italian_sentence.split():
                if word not in self.italian_vocab:
                    self.italian_mapping[word] = italian_counter
                    italian_counter += 1
                    self.italian_vocab.add(word)
    
    def _initialize_translation_probabilities(self) -> List[List[float]]:
        """
        Initialize translation probabilities uniformly.
        
        Returns:
            2D list representing translation probabilities t(f|e)
        """
        english_vocab_size = len(self.english_vocab)
        italian_vocab_size = len(self.italian_vocab)
        
        # Initialize with uniform probabilities
        initial_prob = 1.0 / italian_vocab_size
        translation_probs = [
            [initial_prob] * (italian_vocab_size + 1) 
            for _ in range(english_vocab_size + 1)
        ]
        
        # Set epsilon values for NULL alignments
        for j in self.english_mapping.values():
            translation_probs[j][0] = self.EPSILON
        for i in self.italian_mapping.values():
            translation_probs[0][i] = self.EPSILON
        
        return translation_probs
    
    def _get_word_index(self, word: str, mapping: Dict[str, int]) -> int:
        """Get the index of a word, returning 0 if not found (NULL index)."""
        return mapping.get(word, 0)
    
    def _compute_alignment_probabilities(self, training_data: List[Tuple[str, str]], 
                                        show_progress: bool = False) -> List[List[float]]:
        """
        Compute alignment probabilities for the E-step.
        
        Args:
            training_data: List of (English, Italian) sentence pairs
            show_progress: Whether to show timing and progress information
            
        Returns:
            Alignment probabilities for each sentence pair
        """
        if show_progress:
            start_time = time.time()
            print("Computing alignment probabilities...")
        
        alignment_probs = []
        total_sentences = len(training_data)
        
        for i, (english_sentence, italian_sentence) in enumerate(training_data):
            if show_progress and i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                progress = i / total_sentences
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                print(f"  Progress: {i}/{total_sentences} ({progress:.1%}) - "
                     f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
            
            english_words = english_sentence.split()
            italian_words = italian_sentence.split()
            
            # Compute normalization factors for each Italian word
            normalization_factors = []
            for italian_word in italian_words:
                it_index = self._get_word_index(italian_word, self.italian_mapping)
                total_prob = 0.0
                
                for english_word in english_words:
                    eng_index = self._get_word_index(english_word, self.english_mapping)
                    total_prob += self.translation_probabilities[eng_index][it_index]
                
                normalization_factors.append(total_prob)
            
            alignment_probs.append(normalization_factors)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"  Completed in {total_time:.2f} seconds")
        
        return alignment_probs
    
    def _update_counts(self, training_data: List[Tuple[str, str]], 
                      alignment_probs: List[List[float]], 
                      show_progress: bool = False) -> Tuple[List[List[float]], List[float]]:
        """
        Update count statistics for the M-step.
        
        Args:
            training_data: List of (English, Italian) sentence pairs
            alignment_probs: Precomputed alignment probabilities
            show_progress: Whether to show timing and progress information
            
        Returns:
            Tuple of (count matrix, marginal counts)
        """
        if show_progress:
            start_time = time.time()
            print("Updating count statistics...")
        
        english_vocab_size = len(self.english_vocab)
        italian_vocab_size = len(self.italian_vocab)
        
        # Initialize count matrices
        count = [[0.0] * italian_vocab_size for _ in range(english_vocab_size)]
        single_count = [self.EPSILON] * english_vocab_size
        
        total_sentences = len(training_data)
        
        # Update counts based on alignment probabilities
        for k, (english_sentence, italian_sentence) in enumerate(training_data):
            if show_progress and k % 1000 == 0 and k > 0:
                elapsed = time.time() - start_time
                progress = k / total_sentences
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                print(f"  Progress: {k}/{total_sentences} ({progress:.1%}) - "
                     f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
            
            english_words = english_sentence.split()
            italian_words = italian_sentence.split()
            
            for i, italian_word in enumerate(italian_words):
                it_index = self._get_word_index(italian_word, self.italian_mapping)
                
                for english_word in english_words:
                    eng_index = self._get_word_index(english_word, self.english_mapping)
                    
                    # Compute alignment probability
                    if alignment_probs[k][i] > 0:
                        delta = (self.translation_probabilities[eng_index][it_index] / 
                                alignment_probs[k][i])
                        
                        # Update counts only for words in vocabulary
                        if (english_word in self.english_mapping and 
                            italian_word in self.italian_mapping):
                            count[self.english_mapping[english_word]][self.italian_mapping[italian_word]] += delta
                        
                        if english_word in self.english_mapping:
                            single_count[self.english_mapping[english_word]] += delta
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"  Completed in {total_time:.2f} seconds")
        
        return count, single_count
    
    def _update_translation_probabilities(self, count: List[List[float]], 
                                        single_count: List[float],
                                        show_progress: bool = False) -> None:
        """
        Update translation probabilities based on counts.
        
        Args:
            count: Count matrix for word pairs
            single_count: Marginal counts for English words
            show_progress: Whether to show timing and progress information
        """
        if show_progress:
            start_time = time.time()
        
        for i in range(len(count)):
            for j in range(len(count[0])):
                if single_count[i] > 0:
                    self.translation_probabilities[i][j] = max(
                        count[i][j] / single_count[i], 
                        self.EPSILON
                    )
        
        if show_progress:
            update_time = time.time() - start_time
            print(f"  Translation probabilities updated in {update_time:.3f} seconds")

    def _save_checkpoint(self) -> None:
        """Saves the current iteration and translation probabilities to a file."""
        np.savez(self.CHECKPOINT_FILE, 
                 current_iteration=self.current_iteration, 
                 translation_probabilities=np.array(self.translation_probabilities),
                 english_mapping_keys=list(self.english_mapping.keys()),
                 english_mapping_values=list(self.english_mapping.values()),
                 italian_mapping_keys=list(self.italian_mapping.keys()),
                 italian_mapping_values=list(self.italian_mapping.values()))
        print(f"Checkpoint saved: Iteration {self.current_iteration}, to {self.CHECKPOINT_FILE}")

    def _load_checkpoint(self) -> bool:
        """
        Loads the saved iteration and translation probabilities from a file.
        
        Returns:
            True if a checkpoint was loaded successfully, False otherwise.
        """
        if os.path.exists(self.CHECKPOINT_FILE):
            print(f"Loading checkpoint from {self.CHECKPOINT_FILE}...")
            checkpoint_data = np.load(self.CHECKPOINT_FILE, allow_pickle=True)
            self.current_iteration = checkpoint_data['current_iteration'].item()
            self.translation_probabilities = checkpoint_data['translation_probabilities'].tolist()
            
            # Reconstruct mappings
            self.english_mapping = dict(zip(checkpoint_data['english_mapping_keys'], 
                                            checkpoint_data['english_mapping_values']))
            self.italian_mapping = dict(zip(checkpoint_data['italian_mapping_keys'], 
                                            checkpoint_data['italian_mapping_values']))
            self.english_vocab = set(self.english_mapping.keys())
            self.italian_vocab = set(self.italian_mapping.keys())

            print(f"Checkpoint loaded: Starting from iteration {self.current_iteration + 1}")
            return True
        print("No checkpoint found. Starting training from scratch.")
        return False
    
    def train(self, show_times: bool = False) -> None:
        """
        Train the IBM Model 1 using the EM algorithm.
        
        Args:
            show_times: Whether to show detailed timing information for each function
        """
        # Prepare training data
        training_size = int(self.TRAINING_DATA_PERCENTAGE * len(self.dataset))
        training_data = self.dataset[:training_size]
        
        print(f"Training on {len(training_data)} sentence pairs...")
        
        # Attempt to load a checkpoint
        if self._load_checkpoint():
            start_iteration = self.current_iteration + 1
        else:
            start_iteration = 1
        
        if show_times:
            total_start_time = time.time()
        
        # EM Algorithm iterations
        for iteration in range(start_iteration, self.NUM_ITERATIONS + 1):
            self.current_iteration = iteration
            print(f"\nIteration #{self.current_iteration}")
            
            if show_times:
                iteration_start_time = time.time()
            
            # E-step: Compute alignment probabilities
            if show_times:
                e_step_start = time.time()
            
            alignment_probs = self._compute_alignment_probabilities(training_data, show_progress=show_times)
            
            if show_times:
                e_step_time = time.time() - e_step_start
                print(f"  E-step completed in {e_step_time:.2f} seconds")
            
            # M-step: Update counts
            if show_times:
                m_step_counts_start = time.time()
            
            count, single_count = self._update_counts(training_data, alignment_probs, show_progress=show_times)
            
            if show_times:
                m_step_counts_time = time.time() - m_step_counts_start
                print(f"  M-step (count update) completed in {m_step_counts_time:.2f} seconds")
            
            # M-step: Update probabilities
            if show_times:
                m_step_probs_start = time.time()
            
            self._update_translation_probabilities(count, single_count, show_progress=show_times)
            
            if show_times:
                m_step_probs_time = time.time() - m_step_probs_start
                print(f"  M-step (probability update) completed in {m_step_probs_time:.2f} seconds")
                
                iteration_time = time.time() - iteration_start_time
                print(f"  Total iteration time: {iteration_time:.2f} seconds")
            
            # Save checkpoint after each iteration
            self._save_checkpoint()
        
        if show_times:
            total_training_time = time.time() - total_start_time
            print(f"\nTotal training time: {total_training_time:.2f} seconds")
            print(f"Average time per iteration: {total_training_time / self.NUM_ITERATIONS:.2f} seconds")
    
    def translate(self, source_sentence: str) -> str:
        """
        Translate a source sentence using the learned translation probabilities.
        
        Args:
            source_sentence: English sentence to translate
            
        Returns:
            Translated Italian sentence
        """
        english_words = source_sentence.split()
        translated_words = []
        
        for english_word in english_words:
            eng_index = self._get_word_index(english_word, self.english_mapping)
            
            # Find the Italian word with highest translation probability
            best_italian_word = None
            best_prob = 0.0
            
            # Skip NULL index (0) and search through all Italian words
            for italian_word, it_index in self.italian_mapping.items():
                prob = self.translation_probabilities[eng_index][it_index]
                if prob > best_prob:
                    best_prob = prob
                    best_italian_word = italian_word
            
            # If no translation found or very low probability, keep the original word
            if best_italian_word is None or best_prob < self.EPSILON * 10:
                translated_words.append(english_word)
            else:
                translated_words.append(best_italian_word)
        
        return ' '.join(translated_words)
    
    def translate_batch(self, source_sentences: List[str]) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            source_sentences: List of English sentences to translate
            
        Returns:
            List of translated Italian sentences
        """
        return [self.translate(sentence) for sentence in source_sentences]
    
    def get_best_translations(self, english_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the top-k most likely translations for an English word.
        
        Args:
            english_word: English word to translate
            top_k: Number of top translations to return
            
        Returns:
            List of (italian_word, probability) tuples, sorted by probability (descending)
        """
        if english_word not in self.english_mapping:
            return []
        
        eng_index = self.english_mapping[english_word]
        translations = []
        
        for italian_word, it_index in self.italian_mapping.items():
            prob = self.translation_probabilities[eng_index][it_index]
            translations.append((italian_word, prob))
        
        # Sort by probability (descending) and return top_k
        translations.sort(key=lambda x: x[1], reverse=True)
        return translations[:top_k]
    
    def evaluate_bleu(self, test_sentences: List[Tuple[str, str]], verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate the model using BLEU scores.
        
        Args:
            test_sentences: List of (English, Italian) sentence pairs for evaluation
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary containing BLEU scores and other metrics
        """
        if not test_sentences:
            raise ValueError("No test sentences provided for evaluation")
        
        # Prepare data for evaluation
        source_sentences = [pair[0] for pair in test_sentences]
        reference_sentences = [pair[1] for pair in test_sentences]
        
        # Generate translations
        print("Generating translations for evaluation...")
        translated_sentences = self.translate_batch(source_sentences)
        
        # Prepare references and hypotheses for BLEU calculation
        references_tokenized = [[ref.split()] for ref in reference_sentences]
        hypotheses_tokenized = [hyp.split() for hyp in translated_sentences]
        
        # Calculate BLEU scores with smoothing
        smoothing = SmoothingFunction().method1
        
        # Individual sentence BLEU scores
        sentence_bleu_scores = []
        for i, (ref_tokens, hyp_tokens) in enumerate(zip(references_tokenized, hypotheses_tokenized)):
            try:
                # Calculate BLEU score for each n-gram level
                bleu_1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu_2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu_3 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu_4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                
                sentence_bleu_scores.append({
                    'bleu_1': bleu_1,
                    'bleu_2': bleu_2,
                    'bleu_3': bleu_3,
                    'bleu_4': bleu_4
                })
            except ZeroDivisionError:
                # Handle edge cases where BLEU cannot be calculated
                sentence_bleu_scores.append({
                    'bleu_1': 0.0,
                    'bleu_2': 0.0,
                    'bleu_3': 0.0,
                    'bleu_4': 0.0
                })
        
        # Corpus-level BLEU scores
        try:
            corpus_bleu_1 = corpus_bleu(references_tokenized, hypotheses_tokenized, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            corpus_bleu_2 = corpus_bleu(references_tokenized, hypotheses_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            corpus_bleu_3 = corpus_bleu(references_tokenized, hypotheses_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            corpus_bleu_4 = corpus_bleu(references_tokenized, hypotheses_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        except ZeroDivisionError:
            corpus_bleu_1 = corpus_bleu_2 = corpus_bleu_3 = corpus_bleu_4 = 0.0
        
        # Calculate average sentence-level BLEU scores
        avg_sentence_bleu_1 = sum(score['bleu_1'] for score in sentence_bleu_scores) / len(sentence_bleu_scores)
        avg_sentence_bleu_2 = sum(score['bleu_2'] for score in sentence_bleu_scores) / len(sentence_bleu_scores)
        avg_sentence_bleu_3 = sum(score['bleu_3'] for score in sentence_bleu_scores) / len(sentence_bleu_scores)
        avg_sentence_bleu_4 = sum(score['bleu_4'] for score in sentence_bleu_scores) / len(sentence_bleu_scores)
        
        # Calculate additional metrics
        total_words_translated = sum(len(hyp.split()) for hyp in translated_sentences)
        total_reference_words = sum(len(ref.split()) for ref in reference_sentences)
        
        # Prepare results
        evaluation_results = {
            'corpus_bleu_1': corpus_bleu_1,
            'corpus_bleu_2': corpus_bleu_2,
            'corpus_bleu_3': corpus_bleu_3,
            'corpus_bleu_4': corpus_bleu_4,
            'avg_sentence_bleu_1': avg_sentence_bleu_1,
            'avg_sentence_bleu_2': avg_sentence_bleu_2,
            'avg_sentence_bleu_3': avg_sentence_bleu_3,
            'avg_sentence_bleu_4': avg_sentence_bleu_4,
            'num_test_sentences': len(test_sentences),
            'total_words_translated': total_words_translated,
            'total_reference_words': total_reference_words,
            'avg_sentence_length_translated': total_words_translated / len(test_sentences),
            'avg_sentence_length_reference': total_reference_words / len(test_sentences)
        }
        
        if verbose:
            self._print_evaluation_results(evaluation_results, test_sentences, translated_sentences)
        
        return evaluation_results
    
    def _print_evaluation_results(self, results: Dict[str, float], 
                                 test_sentences: List[Tuple[str, str]], 
                                 translated_sentences: List[str]) -> None:
        """Print detailed evaluation results."""
        print("\n" + "="*60)
        print("BLEU SCORE EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nCorpus-level BLEU Scores:")
        print(f"  BLEU-1: {results['corpus_bleu_1']:.4f}")
        print(f"  BLEU-2: {results['corpus_bleu_2']:.4f}")
        print(f"  BLEU-3: {results['corpus_bleu_3']:.4f}")
        print(f"  BLEU-4: {results['corpus_bleu_4']:.4f}")
        
        print(f"\nAverage Sentence-level BLEU Scores:")
        print(f"  BLEU-1: {results['avg_sentence_bleu_1']:.4f}")
        print(f"  BLEU-2: {results['avg_sentence_bleu_2']:.4f}")
        print(f"  BLEU-3: {results['avg_sentence_bleu_3']:.4f}")
        print(f"  BLEU-4: {results['avg_sentence_bleu_4']:.4f}")
        
        print(f"\nGeneral Statistics:")
        print(f"  Test sentences: {results['num_test_sentences']}")
        print(f"  Average translated sentence length: {results['avg_sentence_length_translated']:.2f} words")
        print(f"  Average reference sentence length: {results['avg_sentence_length_reference']:.2f} words")
        
        print(f"\nSample Translations:")
        print("-" * 60)
        for i in range(min(5, len(test_sentences))):
            print(f"Source:    {test_sentences[i][0]}")
            print(f"Reference: {test_sentences[i][1]}")
            print(f"Generated: {translated_sentences[i]}")
            print("-" * 60)


def main():
    """Main function to demonstrate IBM Model 1 usage with evaluation."""
    print("Initializing IBM Model 1...")
    model = IBMModel1()
    
    print(f"Italian Vocabulary Size: {len(model.italian_vocab)}")
    print(f"English Vocabulary Size: {len(model.english_vocab)}")
    print(f"Translation Probabilities Shape: {len(model.translation_probabilities)} x {len(model.translation_probabilities[0])}")
    
    # Train the model with timing information
    # Set show_times=True to see detailed timing information
    model.train(show_times=True)
    
    # Prepare test data (use remaining data for testing)
    training_size = int(model.TRAINING_DATA_PERCENTAGE * len(model.dataset))
    test_data = model.dataset[training_size:training_size + 5]  # Use 500 sentences for testing
    
    if test_data:
        print(f"\nEvaluating model on {len(test_data)} test sentences...")
        evaluation_results = model.evaluate_bleu(test_data)
        
        # Save evaluation results
        print("\nEvaluation complete!")
        print(f"Main BLEU-4 Score: {evaluation_results['corpus_bleu_4']:.4f}")
    else:
        print("\nNo test data available for evaluation.")
    
    # Demonstrate individual word translation
    print("\nExample word translations:")
    test_words = ["hello", "the", "house", "good", "time"]
    for word in test_words:
        if word in model.english_mapping:
            best_translations = model.get_best_translations(word, top_k=3)
            print(f"{word} -> {best_translations}")
    
    # Demonstrate sentence translation
    print("\nExample sentence translations:")
    test_sentences = [
        "the house is good",
        "hello my friend",
        "this is a test"
    ]
    
    for sentence in test_sentences:
        translation = model.translate(sentence)
        print(f"EN: {sentence}")
        print(f"IT: {translation}")
        print("-" * 40)


if __name__ == "__main__":
    main()