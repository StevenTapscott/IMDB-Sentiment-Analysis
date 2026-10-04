from transformers import pipeline
#from datasets import load_dataset
#import pandas as pd
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load a pre-trained sentiment analysis model
print("Loading...")
classifier = pipeline("sentiment-analysis", 
model = "distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded sucessfully!\n")

#Interactive testing
def analyse_custom_text():
    print("\n" + "="*50)
    print("Custom Sentiment Analysis")
    print("="*50)
    print("Enter movie reviews to analyse (or quit to exit)\n")

    while True:
        user_input = input("Enter review: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            result = classifier(user_input)[0]
            print(f"Sentiment: {result['label']}")
            print(f"Confidence: {result['score']:.2%}\n")

if __name__ == "__main__":
    analyse_custom_text()



#Load IMDB Dataset (Code below used for testing the model)
#print("Loading Dataset...")
#dataset = load_dataset("stanfordnlp/imdb", split="test") 
#dataset = dataset.shuffle(seed=42)

#print(f"Dataset size: {len(dataset)}")

#Function to predict sentiment for evaluation
#def predict_sentiment(texts, batch_size=32):
    #predictions = []
    #for i in range(0, len(texts), batch_size):
        #batch = texts[i:i+batch_size]
        #print(
            #f"Processing {i + 1}-"
            #f"{min(i + batch_size, len(texts))} "
            #f"of {len(texts)}"
        #)
        #results = classifier(batch, truncation=True, max_length=512)
        #predictions.extend(results)
    #return predictions

#texts = dataset['text']

#Get predictions
#print("Making predictions...")
#texts = dataset['text']
#predictions = predict_sentiment(texts)

#Convert predictions to Binary (1 = Positive, 0 = Negative)
#pred_labels = [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
#true_labels = dataset['label']

#Evaluate the model accuracy
#accuracy = accuracy_score(true_labels, pred_labels)
#print(f"\nAccuracy: {accuracy:.4f}")
#print("\nClassification Report:")
#print(classification_report(true_labels, pred_labels,
#target_names=['Negative', 'Positive']))

# Some examples
#print("\n--- Example Predictions ---")
#for i in range(5):
    #print(f"\nReview: {texts[i][:200]}...")
    #print(f"True: {'Positive' if true_labels[i] == 1 else 'Negative'}")
    #print(f"Predicted: {predictions[i]['label']} (confidence: {predictions[i]['score']:.3f})")


#(Training purposes)Test it with a simple example (text = "This movie was fantastic! I loved every minute.")
#text = """The story of a reluctant Christ-like protagonist set against a baroque, MTV backdrop, The Matrix is the definitive hybrid of technical wizardry and contextual excellence that should be the benchmark for all sci-fi films to come.

#Hollywood has had some problems combining form and matter in the sci-fi genre. There have been a lot of visually stunning works but nobody cared about the hero. (Or nobody simply cared about anything.) There a few, though, which aroused interest and intellect but nobody 'ooh'-ed or 'aah'-ed at the special effects. With The Matrix, both elements are perfectly en sync. Not only did we want to cheer on the heroes to victory, we wanted them to bludgeon the opposition. Not only did we sit in awe as Neo evaded those bullets in limbo-rock fashion, we salivated.

#But what makes The Matrix several cuts above the rest of the films in its genre is that there are simply no loopholes. The script, written by the Wachowski brothers is intelligent but carefully not geeky. The kung-fu sequences were deftly shot *--* something even Bruce Lee would've been proud of. The photography was breathtaking. (I bet if you had to cut every frame on the reel and had it developed and printed, every single frame would stand on its own.) And the acting? Maybe not the best Keanu Reeves but name me an actor who has box-office appeal but could portray the uneasy and vulnerable protagonist, Neo, to a T the way Reeves did. But, come to think of it, if you pit any actor beside Laurence Fishburne, you're bound to confuse that actor for bad acting. As Morpheus, Mr. Fishburne is simply wicked! Shades of his mentor-role in Higher Learning, nobody exudes that aura of quiet intensity than Mr. Fishburne. His character, battle-scarred but always composed Morpheus, is given an extra dose of mortality (He loves Neo to a fault.) only Mr. Fishburne can flesh out.

#People will say what they want to say about how good The Matrix is but the bottomline is this: finally there's a philosophical film that has cut through this generation. My generation. The Wachowski brothers probably scribbled a little P.S. note when they finished the script saying: THINK FOR A MOMENT ABOUT YOUR EXISTENCE. What is the Matrix, you ask? Something that's closer to reality than you think. Either that or it's my personal choice for best film of all-time."""

#result = classifer(text, truncation=True, max_length=512)

#print(f"Text: {text}")
#print(f"Result: {result}")


