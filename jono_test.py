# !pip install sentence-transformers einops timm pillow
from sentence_transformers import SentenceTransformer

# Choose a matryoshka dimension
truncate_dim = 512

# Initialize the model
model = SentenceTransformer(
    '/mnt/e/hf/jina-clip-v2', trust_remote_code=True, truncate_dim=truncate_dim
)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

# Encode text and images
text_embeddings = model.encode(sentences, normalize_embeddings=True)
image_embeddings = model.encode(
    image_urls, normalize_embeddings=True
)  # also accepts PIL.Image.Image, local filenames, dataURI

# Encode query text
query = 'beautiful sunset over the beach' # English
query_embeddings = model.encode(
    query, prompt_name='retrieval.query', normalize_embeddings=True
)
