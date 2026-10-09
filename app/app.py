import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "sonia250/afriberta-kirundi-news"

st.set_page_config(page_title="Kirundi News Classifier", page_icon="📰")


@st.cache_resource(show_spinner="Loading model...")
def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, low_cpu_mem_usage=True)
    model.eval()
    return tok, model


tok, model = load()

st.title("Kirundi News Topic Classifier")
st.write(
    "A fine-tuned AfriBERTa model that classifies Kirundi news into 6 topics: "
    "business, entertainment, health, politics, religion, sports. "
    "Trained on the MasakhaNEWS Kirundi dataset. "
    "Longer articles give more reliable results."
)

EXAMPLES = {
    "Sports headline": "Chelsea: Roman Abramovich avuga ko agiye kugurisha uyu murwi",
    "Health headline": "Covid: Iran yatanguye kugerageza urucanco rwayo",
    "Politics headline": "Ukraine: Mbega Putin ashaka iki, ubwo Russia izoteba ihagarika intambara?",
    "Religion headline": "Uganda: Musenyeri mukuru wa Kampala Cyprian Kizito Lwanga yitavye Imana",
    "Business headline": "Perezida Nkurunziza avuga ko u Burundi bugiye gutangura kwimba Coltan kuva 2020",
    "Entertainment (headline + summary)": "Video: Mu munota umwe... ubuzima bwa Queen Elizabeth II. Ubwami bw'Ubwongereza bwabuze Umwamikazi, Elizabeth II, yari amaze imyaka 70 ku ngoma. Ubu ni ubuzima bwiwe kuva yimikwa gushika ku munsi wiwe wanyuma... mu munota umwe.",
    "Politics (headline + summary)": "Abanye Ethiopia bavuga igituma bashaka kuja muri Russia. Ababoneka batonze imirongo imbere y'ubuserukizi bwa Russia muri Ethiopia, bavuga ko bashaka kuja kurwana muri Ukraine, cane cane kuko ubuzima bugoye mu gihugu.",
}
st.caption("Tip: paste the headline AND the article. A headline alone is much less reliable.")
choice = st.selectbox("Try an example (optional)", ["(write my own)"] + list(EXAMPLES))
default = "" if choice == "(write my own)" else EXAMPLES[choice]
text = st.text_area("Kirundi news text (headline, or headline + article)",
                    value=default, height=220)

if st.button("Classify"):
    if len(text.split()) < 3:
        st.warning("Please enter at least a short Kirundi headline or article.")
    else:
        inputs = tok(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        results = sorted(
            ((model.config.id2label[i], float(p)) for i, p in enumerate(probs)),
            key=lambda x: -x[1])
        st.success(f"Predicted topic: **{results[0][0]}** ({results[0][1]:.1%})")
        st.bar_chart(pd.DataFrame(
            {"probability": [p for _, p in results]},
            index=[c for c, _ in results]))