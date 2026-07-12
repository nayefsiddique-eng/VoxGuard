import random

CHALLENGE_POOL = [
    # Digit sequence challenges (tests sequential processing lag and number modeling)
    "Please repeat the numbers: 8, 3, 9, 1, 4.",
    "Please repeat the numbers: 7, 2, 0, 5, 6.",
    # Acoustic modulation challenges (tests ability to mimic atypical vocalizations)
    "Please whisper: 'The quick brown fox jumps over the lazy dog'.",
    "Please laugh loudly for two seconds.",
    "Please hum a simple three-note melody.",
    # Linguistic spontaneity challenges (tests real-time synthesis speed vs response latency)
    "What is 15 plus 27? Tell me the answer.",
    "What color is a banana? Answer immediately.",
    "Please spell the word 'VOICE' backwards."
]

def generate_challenge_phrase():
    """Generates a random challenge prompt to verify caller authenticity."""
    return random.choice(CHALLENGE_POOL)

if __name__ == "__main__":
    print("Testing Challenge Generation:")
    for i in range(3):
        print(f"  Challenge {i+1}: {generate_challenge_phrase()}")
