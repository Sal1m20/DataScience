import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Charger le tokenizer et le modèle BART pré-entraîné
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Fonction pour générer du texte avec BART
def generer_texte(texte_entree, max_length=100, num_beams=5):
    inputs = tokenizer(texte_entree, return_tensors="pt", max_length=max_length, truncation=True)
    out = model.generate(**inputs, max_length=max_length, num_beams=num_beams, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Interface utilisateur Streamlit
def main():
    st.title("Présentation du modèle Transformer-BART")
    st.write("L'objectif de cette présentation est d'introduire le modèle Transformer-BART et son application à la synthèse de texte abstraite.")

    # Zone de texte pour l'utilisateur saisit son texte
    texte_entree = st.text_area("Entrez votre texte :")

    # Bouton pour générer du texte
    if st.button("Générer Texte"):
        if texte_entree:
            texte_genere = generer_texte(texte_entree)
            st.subheader("Texte Généré :")
            st.write(texte_genere)

if __name__ == "__main__":
    main()
