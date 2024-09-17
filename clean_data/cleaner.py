#The cleaner script defines the cleaning function that should be applied to the bodies of the scrapped news to get the desired clean
# bodies. Those clean bodies are more suitable to be used as inputs for a DeepLearning model.
import re
import nltk

nltk.download('punkt')

#Define common promotional phrases
promo_phrases = [
    "GO AD FREE", "Get your Digital Subscription", "Use code", "More on", "News you can trust",
    "day forecast", "Home", "Article", "click here", "buy now", "limited time offer", "subscribe now",
    "unlock the no-ads mode", "go premium", "ad-free", "Free trial", "Sign up now", "Limited time offer",
    "Act now", "Don't miss out", "Buy one get one free", "Special discount", "Hurry up", "Order now",
    "Money back guarantee", "Exclusive offer", "Best deal", "Save now", "Get started", "Shop now",
    "Learn more", "Subscribe today", "Best price", "Offer ends soon", "Download now", "Call now",
    "Visit our website", "Premium access", "Instant access", "Join now", "Unlock access", "VIP access",
    "Discount code", "Promo code", "Register now", "Claim your free gift", "Satisfaction guaranteed",
    "Follow us", "Connect with us", "Like and share", "Follow for updates", "Upgrade now", "Become a member",
    "Sign in to access", "For more information", "Read more", "Click to continue", "Watch now", "Find out more",
    "Discover more", "Start your journey", "Enhance your experience", "Get the best", "Reserve your spot",
    "Act fast", "Check this out", "Try it free", "Explore now", "Book your place", "Request a quote",
    "Limited stock", "On sale now", "Contact us today", "Free shipping", "Subscribe for updates",
    "Click to learn more", "Experience the best", "Shop the collection", "Buy now pay later", "Email newsletter"
    "Limited time only", "Get yours today"
]

class Cleaner:
    """
    A class used to clean text raw data from promotional content and other unwanted characters.
    It is intended to be used with the scraped news from GDELT.
    
    Attributes
    ----------
    max_length : int
        Maximum length of the text after cleaning.
    min_length : int
        Minimum length of the text after cleaning.
    
    Methods
    -------
    clean_text(text)
        Cleans the provided text according to the specified rules.
    """
    def __init__(self, max_length=10000, min_length=500):
        """
        Parameters
        ----------
        max_length : int, optional
            Maximum length of the text after cleaning (default is 10000).
        min_length : int, optional
            Minimum length of the text after cleaning (default is 500).
        """
        self.max_length = max_length
        self.min_length = min_length

    def clean_text(self, text):
        """
        Cleans the provided text according to the specified rules.
        
        Parameters
        ----------
        text : str
            The text to be cleaned.
        
        Returns
        -------
        str or None
            The cleaned text if it meets the length requirements, otherwise None.
        
        Raises
        ------
        TypeError
            If the provided text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("The input text must be a string.")

        try:
            #Remove non-printable characters
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            
            #Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            #Remove special symbols (keeping regular punctuation)
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"]+', '', text)
            
            #Remove double .. that may have been generated because of the way the lambda function is defined
            text = text.replace("..", ".")

            #Split the text into sentences (punctuation is preserved)
            sentences = nltk.sent_tokenize(text)

            #Remove sentences containing promotional phrases
            cleaned_sentences = [sentence for sentence in sentences if not any(promo.lower() in sentence.lower() for promo in promo_phrases)]

            #Join the cleaned sentences back into a single text
            text = ' '.join(cleaned_sentences)
            
            #Enforce min and max length
            if len(text) < self.min_length or len(text) > self.max_length:
                return None
            
            return text
        except Exception as e:
            print(f"An error occurred while cleaning the text: {e}")
            return None