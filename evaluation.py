import json
import numpy as np

# S-BERT Evaluater
from sentence_transformers import SentenceTransformer, util as SentenceTransformerUtil
from sklearn.metrics.pairwise import cosine_similarity

# JSD
from scipy.spatial.distance import jensenshannon
from collections import Counter

# BERTScore
from bert_score import score as calculate_bert_score

# Visualization
import matplotlib.pyplot as plt

# KLD
from collections import Counter
from scipy.stats import entropy

LABELS = [
    "Gemini 3 Pro Reference Text",
    "Spam Template",# \n(Spammimic)
    "Custom Template",# \n(Stegosaurus)
    "White Space Unicode",# \n(SNOW)
    "Zero-Width Unicode",# \n(StegCloak)
    "Variation Selectors",
    "Homoglyphes",# \n(StegText)
    "State Transferring Probability",# \n(plainsight)
    "LLM Generated with Context",# (Stegasuras)
    "Persian words",# \n(NAHOFT)
]
COLORS = [
    "tab:blue",
    "tab:purple",
    "tab:pink",
    "tab:green",
    "tab:brown",
    "gold",
    "tab:olive",
    "tab:orange",
    "tab:red",
    "tab:cyan"
]
FIGURE_SIZE = (8, 5)

class EvaluationHandler():

    def run():
        # 32 byte secret
        secret = "This is a very secret message!!!"

        chat_histories = json.load(open('chat_histories.json', 'r'))
        stego_texts = json.load(open('stego_texts.json', 'r'))
        all_results = {}

        # Iterate through each tool and run evaluation
        for tool in stego_texts:
            tool_name = tool['tool']
            stego_text_list = tool['stego_text']

            evaluator = Evaluater(chat_histories, stego_text_list)
            results = evaluator.run_evaluation()
            all_results[tool_name] = results
            
            # Print results
            print(f"--- Evaluation Results for {tool_name} ---")
            for idx, metrics in results.items():
                print(f"Chat {idx}:")
                for metric_name, score in metrics.items():
                    print(f"  {metric_name}: {score:.4f}")
            print("\n")
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)

        # Create plots
        EvaluationHandler._create_plots()
    
    @staticmethod
    def _calculate_bits_per_character():
        '''
        Calculates and prints the Bits Per Character (BPC) for each steganography tool.
        BPC is computed as the average number of embedded bits divided by the average length
        of the stego texts produced by each tool.

        Returns:
            dict: A dictionary with tool names as keys and their corresponding BPC values.
        '''
        bpc_results = {}
        with open('stego_texts.json', 'r') as f:
            stego_texts = json.load(f)
        for tool in stego_texts:
            print(f"Tool: {tool['tool']}")
            avg_embedded_bits = tool['embedded-bits']
            avg_len = np.mean([len(text) for text in tool['stego_text']])
            bpc = avg_embedded_bits / avg_len
            print(f"  Avg Length: {avg_len:.2f} characters")
            print(f"  Bits per Character: {bpc:.4f} bits/character\n")
            bpc_results[tool['tool']] = bpc
        return bpc_results


    @staticmethod
    def _create_plots():
        # Load evaluation results
        all_results = json.load(open('evaluation_results.json', 'r'))
        stego_texts = json.load(open('stego_texts.json', 'r'))

        # Create plots for each metric
        EvaluationHandler._create_jsd_plot(all_results, stego_texts)
        EvaluationHandler._create_mean_jsd_plot(all_results, stego_texts)
        EvaluationHandler._create_kld_plot(all_results, stego_texts)
        EvaluationHandler._create_bert_score_plot(all_results, stego_texts)
        EvaluationHandler._create_mean_bert_score_plot(all_results, stego_texts)
        EvaluationHandler._create_s_bert_plot(all_results, stego_texts)
        EvaluationHandler._create_mean_s_bert_plot(all_results, stego_texts)
        EvaluationHandler._create_bpc_plot(stego_texts)

    @staticmethod
    def _create_mean_s_bert_plot(all_results, stego_texts):
        global LABELS
        s_bert_scores = []

        # Calculate mean and stddev for each tool
        for _, value in all_results.items():
            s_bert_scores.append([results['s_bert'] for results in value.values()])
        vector = np.array(s_bert_scores)
        vector_mean = np.mean(vector, axis=1)
        vector_std = np.std(vector, axis=1)

        # Sort by mean score
        data_labels = {label: [mean, srd, stego_text['reference'], stego_text['gls']] for label, mean, srd, stego_text in zip(LABELS, vector_mean, vector_std, stego_texts)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
        values = list(data_labels.values())
        labels = list(data_labels.keys())

        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        x = np.arange(len(labels))
        for i, value in enumerate(values):
            if value[2] or value[3]:
                plt.bar(x[i], values[i][0], yerr=values[i][1], capsize=5, color='#629FCA')
            else:
                plt.bar(x[i], values[i][0], yerr=values[i][1], capsize=5, color='lightgray')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Mean S-BERT Score', fontsize=12)
        #plt.title('Mean S-BERT Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_s_bert_mean.png', dpi=300, bbox_inches='tight')


    @staticmethod
    def _create_s_bert_plot(all_results, stego_texts):
        global LABELS
        global COLORS
        s_bert_scores = []

        # Calculate mean for each tool
        for _, value in all_results.items():
            s_bert_scores.append([results['s_bert'] for results in value.values()])
        vector = np.array(s_bert_scores)
        vector_mean = np.mean(vector, axis=1)

        # Sort by bertscore mean
        data_labels = {label: [mean, result, stego_text['reference'], stego_text['gls'], color] for label, mean, result, stego_text, color in zip(LABELS, vector_mean, s_bert_scores, stego_texts, COLORS)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
            
        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        for label, (mean, scores, reference, gls, color) in data_labels.items():
            valid_indices = range(len(scores))
            
            if reference:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=8,
                        color=color)
            else:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='--',
                        linewidth=1.5,
                        marker='s',
                        markersize=5,
                        color=color)
        
        plt.xlabel('Chat History Index', fontsize=12)
        plt.ylabel('S-BERT Score', fontsize=12)
        #plt.title('S-BERT Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_s_bert.png', dpi=300, bbox_inches='tight')


    @staticmethod
    def _create_bert_score_plot(all_results, stego_texts):
        global LABELS
        global COLORS
        bert_score_scores = []

        # Calculate mean for each tool
        for _, value in all_results.items():
            bert_score_scores.append([results['bert_score'] for results in value.values()])
        vector = np.array(bert_score_scores)
        vector_mean = np.mean(vector, axis=1)

        # Sort by bertscore mean
        data_labels = {label: [mean, result, stego_text['reference'], stego_text['gls'], color] for label, mean, result, stego_text, color in zip(LABELS, vector_mean, bert_score_scores, stego_texts, COLORS)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
            
        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        for label, (mean, scores, reference, gls, color) in data_labels.items():
            valid_indices = range(len(scores))
            
            if gls:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='--',
                        linewidth=1.5,
                        marker='s',
                        markersize=5,
                        color='lightgray')
            elif reference:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=8,
                        color=color)
            else:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='--',
                        linewidth=1.5,
                        marker='s',
                        markersize=5,
                        color=color)
        
        plt.xlabel('Chat History Index', fontsize=12)
        plt.ylabel('BERTScore', fontsize=12)
        #plt.title('BERTScore Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_bert_score.png', dpi=300, bbox_inches='tight')

    @staticmethod
    def _create_mean_bert_score_plot(all_results, stego_texts):
        global LABELS
        bertscore_scores = []

        # Calculate mean and stddev for each tool
        for _, value in all_results.items():
            bertscore_scores.append([results['bert_score'] for results in value.values()])
        vector = np.array(bertscore_scores)
        vector_mean = np.mean(vector, axis=1)
        vector_std = np.std(vector, axis=1)

        # Sort by mean score
        data_labels = {label: [mean, srd, stego_text['reference'], stego_text['gls']] for label, mean, srd, stego_text in zip(LABELS, vector_mean, vector_std, stego_texts)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
        values = list(data_labels.values())
        labels = list(data_labels.keys())

        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        x = np.arange(len(labels))
        for i, value in enumerate(values):
            if not value[3]:
                plt.bar(x[i], values[i][0], yerr=values[i][1], capsize=5, color='#629FCA')
            else:
                plt.bar(x[i], values[i][0], yerr=values[i][1], capsize=5, color='lightgray')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Mean BERETScore', fontsize=12)
        #plt.title('Mean S-BERT Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_bert_score_mean.png', dpi=300, bbox_inches='tight')

    @staticmethod
    def _create_jsd_plot(all_results, stego_texts):
        global LABELS
        global COLORS
        jsd_scores = []

        # Calculate mean for each tool
        for _, value in all_results.items():
            jsd_scores.append([results['jsd'] for results in value.values()])
        vector = np.array(jsd_scores)
        vector_mean = np.mean(vector, axis=1)

        # Sort by bertscore mean
        data_labels = {label: [mean, result, stego_text['reference'], stego_text['gls'], color] for label, mean, result, stego_text, color in zip(LABELS, vector_mean, jsd_scores, stego_texts, COLORS)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
        
        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        for label, (mean, scores, reference, gls, color) in data_labels.items():
            valid_indices = range(len(scores))
            
            if reference:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=8,
                        color=color)
            else:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='--',
                        linewidth=1.5,
                        marker='s',
                        markersize=5,
                        color=color)
        
        plt.xlabel('Chat History Index', fontsize=12)
        plt.ylabel('JSD Score', fontsize=12)
        #plt.title('JSD Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_jsd.png', dpi=300, bbox_inches='tight')

    @staticmethod
    def _create_mean_jsd_plot(all_results, stego_texts):
        global LABELS
        jsd_scores = []

        # Calculate mean and stddev for each tool
        for _, value in all_results.items():
            jsd_scores.append([results['jsd'] for results in value.values()])
        vector = np.array(jsd_scores)
        vector_mean = np.mean(vector, axis=1)
        vector_std = np.std(vector, axis=1)

        # Sort by mean score
        data_labels = {label: [mean, srd, stego_text['reference'], stego_text['gls']] for label, mean, srd, stego_text in zip(LABELS, vector_mean, vector_std, stego_texts)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
        values = list(data_labels.values())
        labels = list(data_labels.keys())

        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        x = np.arange(len(labels))
        for i, value in enumerate(values):
            plt.bar(x[i], values[i][0], yerr=values[i][1], capsize=5, color='#629FCA')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Mean JSD Score', fontsize=12)
        #plt.title('Mean S-BERT Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_jsd_mean.png', dpi=300, bbox_inches='tight')

    @staticmethod
    def _create_kld_plot(all_results, stego_texts):
        global LABELS
        global COLORS
        kld_scores = []

        # Calculate mean for each tool
        for _, value in all_results.items():
            kld_scores.append([results['kld'] for results in value.values()])
        vector = np.array(kld_scores)
        vector_mean = np.mean(vector, axis=1)

        # Sort by bertscore mean
        data_labels = {label: [mean, result, stego_text['reference'], stego_text['gls'], color] for label, mean, result, stego_text, color in zip(LABELS, vector_mean, kld_scores, stego_texts, COLORS)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
            
        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        for label, (mean, scores, reference, gls, color) in data_labels.items():
            valid_indices = range(len(scores))
            
            if reference:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=8,
                        color=color)
            else:
                plt.plot(valid_indices, 
                        scores, 
                        label=label,
                        linestyle='--',
                        linewidth=1.5,
                        marker='s',
                        markersize=5,
                        color=color)
        
        plt.xlabel('Chat History Index', fontsize=12)
        plt.ylabel('KLD Score', fontsize=12)
        #plt.title('KLD Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_kld.png', dpi=300, bbox_inches='tight')


    @staticmethod
    def _create_bpc_plot(stego_texts):
        global LABELS
        bpc_results = EvaluationHandler._calculate_bits_per_character()

        # Sort by mean score
        data_labels = {label: [bpc, stego_text['reference'], stego_text['gls']] for label, bpc, stego_text in zip(LABELS, bpc_results.values(), stego_texts)}
        data_labels = dict(sorted(data_labels.items(), key=lambda item: item[1][0], reverse=True))
        data_labels = dict(filter(lambda item: not item[1][1], data_labels.items()))  # Filter out reference tool
        values = list(data_labels.values())
        labels = list(data_labels.keys())

        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        x = np.arange(len(labels))
        for i, value in enumerate(values):
            plt.bar(x[i], values[i][0], capsize=5, color='#629FCA')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Bits Per Character', fontsize=12)
        #plt.title('Bits Per Character Evaluation Results Across Tools', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_bpc.png', dpi=300, bbox_inches='tight')


class Evaluater:

    histories: dict = []

    def __init__(self, chat_histories: list[list[str]], stego_texts: list[str]):
        self.chat_histories = chat_histories
        self.stego_texts = stego_texts
    
    def get_histories(self) -> list[list[str]]:
        return self.chat_histories

    def get_stego_texts(self) -> list[str]:
        return self.stego_texts
    
    def get_context_window(self) -> int:
        return min([len(history) for history in self.get_histories()])
    
    def run_evaluation(self) -> dict[str, float]:
        # Run evaluation for each chat history and stego text
        evaluation_results = {i: None for i in range(0,len(self.get_histories()))}
        context_window = self.get_context_window()
        for i, chat_history in enumerate(self.get_histories()):
            chat_history = chat_history[-context_window:-1]
            cover_text = chat_history[-1]
            stego_text = self.get_stego_texts()[i]
            evaluation_results[i] = {
                "s_bert": (SBertEvaluater(chat_history, stego_text, cover_text)).run(),
                "jsd": (JSDEvaluater(chat_history, stego_text, cover_text)).run(),
                "kld": (KLDEvaluater(chat_history, stego_text, cover_text)).run(),
                "bert_score": (BertScoreEvaluater(chat_history, stego_text, cover_text)).run() if cover_text is not None else None
            }
        return evaluation_results


class EvaluaterMetric:

    def __init__(self, chat_history: list[str], stego_text: str, cover_text: str = None):
        self.chat_history = chat_history
        self.stego_text = stego_text
        self.cover_text = cover_text
        
    def run(self):
        pass

    def get_chat_history(self) -> list[str]:
        return [message[10:] for message in self.chat_history]

    def get_chat_history_string(self) -> str:
        return "\n".join(self.get_chat_history()) 
    
    def get_stego_text(self) -> str:
        return self.stego_text[10:]
    
    def get_cover_text(self) -> str:
        return self.cover_text[10:] if self.cover_text is not None else None


class KLDEvaluater(EvaluaterMetric):

    def run(self):
        """
        Calculates the Kullback-Leibler Divergence (KLD) of the Stego Text 
        relative to the Chat History.
            
        Returns:
            float: The KLD score. 
                Lower (near 0) = Matches history statistically.
                Higher = Statistical Anomaly.
        """
        
        # Count Frequencies (P and Q)
        # We treat Stego as 'P' (The observed distribution we want to test)
        # We treat History as 'Q' (The expected/reference distribution)
        p_counts = Counter(self.get_stego_text())
        q_counts = Counter(self.get_chat_history_string())
        
        # Create the "Union" Alphabet
        # We must evaluate over all characters present in EITHER text.
        union_alphabet = set(p_counts.keys()) | set(q_counts.keys())
        sorted_vocab = sorted(list(union_alphabet))
        
        # Create Probability Vectors with Smoothing
        # We add 'smoothing_alpha' to every count. This ensures that even if 
        # a character is missing in the history, it has a prob > 0.
        smoothing_alpha = 1e-15
        P = np.array([p_counts[char] + smoothing_alpha for char in sorted_vocab], dtype=float)
        Q = np.array([q_counts[char] + smoothing_alpha for char in sorted_vocab], dtype=float)
        
        # Normalize to create valid Probability Distributions (Sum = 1)
        P = P / P.sum()
        Q = Q / Q.sum()
        
        # Calculate KLD
        # scipy.stats.entropy(pk, qk) computes S = sum(pk * log(pk / qk))
        kl_score = entropy(P, Q)
        
        return kl_score


class JSDEvaluater(EvaluaterMetric):
    
    def run(self) -> float:
        """
        Calculates the Jensen-Shannon Divergence (JSD) between a chat history 
        and a stego text.
            
        Returns:
            float: The JSD score (0.0 to 1.0). 
                0.0 = Identical distributions.
                1.0 = Completely disjoint distributions (no shared characters).
        """
        
        # Count Frequencies (Character-Level)
        p_counts = Counter(self.get_chat_history_string())
        q_counts = Counter(self.get_stego_text())
        
        # Create the "Union" Alphabet
        # JSD requires vectors of the same length. We define the alphabet as 
        # every unique character appearing in EITHER the history OR the stego text.
        union_alphabet = set(p_counts.keys()) | set(q_counts.keys())
        sorted_vocab = sorted(list(union_alphabet))
        
        # Create Probability Vectors
        # We map the counts to the sorted vocabulary. If a character is missing
        # in one distribution (e.g., an emoji only in the stego text), 
        # it gets a count of 0.
        P = np.array([p_counts[char] for char in sorted_vocab], dtype=float)
        Q = np.array([q_counts[char] for char in sorted_vocab], dtype=float)
            
        # Calculate JSD
        # scipy.spatial.distance.jensenshannon computes the "Jensen-Shannon Distance"
        # which is the square root of the Divergence.
        # Typically, the Metric JSD is the square of the distance.
        js_distance = jensenshannon(P, Q, base=2.0)
        js_divergence = js_distance ** 2
        
        return js_divergence


class SBertEvaluater(EvaluaterMetric):

    def run(self) -> float:
        """
        Calculates semantic similarity between a single chat history and a stego_text.
            
        Returns:
            A single float score (0.0 to 1.0).
        """
        
        # Load the SBERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Compute Embeddings
        embedding_history = model.encode(self.get_chat_history_string(), convert_to_tensor=True)
        embedding_stego = model.encode(self.get_stego_text(), convert_to_tensor=True)

        # Calculate Cosine Similarity
        score = SentenceTransformerUtil.cos_sim(embedding_history, embedding_stego)

        # Extract the single float value from the tensor
        return score.item()


class BertScoreEvaluater(EvaluaterMetric):

    def run(self) -> float:
        """
        Calculates the Semantic Fidelity of a stego-text relative to its cover-text
        using BERTScore.

        Returns:
            float: The F1 BERTScore (0.0 to 1.0). 
                Higher score = Higher fidelity (meaning is better preserved).
        """
        
        # Prepare Inputs
        # BERTScore expects lists of strings. 
        # We treat the inputs as a single pair of sentences/contexts.
        cands = [self.get_stego_text()]
        refs = [self.get_cover_text()]

        # Calculate BERTScore
        P, R, F1 = calculate_bert_score(cands, refs, lang="en", verbose=False)

        # 3. Extract Metric
        # The library returns PyTorch tensors. We extract the scalar F1 value.
        # F1 is the harmonic mean of Precision (P) and Recall (R).
        fidelity_score = F1.mean().item()
        
        return fidelity_score


if __name__ == "__main__":
    EvaluationHandler._create_plots()
