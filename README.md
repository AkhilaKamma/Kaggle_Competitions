**NLP with TensorFlow**
Tensorflow is a powerful Machine Lerning library that was developed by Google Brain team. You can use it for all sorts of tasks from Image classification, all the way to NLP.
what is NLP?
- Natural Language Processing or NLP is an area of research and computer science focusing on processing natural languages, such as English and German and so on. Now, what do we mean by processing? Well, we're taking these languages and converting them into numbers that a computer understands. Computers then perform several tasks, such as text classification or text generation, or answering questions to demonstrate this understanding
- Tasks in NLP
   1. Sentence Classification ex,Sentiment analysis
   2. Named Entity recognition
   3. Auto Generated Text
   4. Text Translation or Text Summarization
   5. Question and Answering
- Google search uses BERT, it capture the importance of nuances.
- Model evolutions
    1. In June, 2018, GPT or generative pre-training model, which was developed by open AI was the first pre-train transformer model and was used for fine tuning on various NRP tasks and obtained state of the art results.
    2. Later that year in October, 2018, researchers at Google came up with BERT or bi-directional encoder representations from transformers.
    3.  In February, 2019, open AI released a bigger and better version of GPT called the GPT-2.
    4.  Later that year in October, 2019, Facebook's AI research team released BART, or bi-directional and auto regressive transformer and Google released T5. Both of these models are larger pre-trained models using the same architecture as the original transformer.
    5.  In the same month, the team at Hugging Face bucked the trend. Everyone was moving to bigger models. The Hugging Face team released DistilBERT, which is a smaller, faster and lighter version of BERT and has 95% of BERT performance on the GLUE language understanding benchmark. And you can see that over the years, the trend is to have bigger model sizes.
**Bias in BERT and GPT**
-This man works as a ----- Bert outputs the results as [carpenter,doctor, farmer and business]
-This woman works as a ------ Bert outputs the results as [nurse,teacher,model,lawyer]
- I've also found occasional examples of bias against gay people with the BERT model. Now, if you do some more study, you might also find bias, like the jobs for men might be for a doctor, but the job for a woman might be a nurse and not a doctor.
- Now, with GPT-2, in this case, each time you run the model, you'll get a different auto-generated text. any significant bias against men, women, white, or black. However, again, I found examples of bias against gay people in GPT-2. And additionally, I found many examples of racism and bias against people from the Middle East. So auto-generated text had references to antisemitism, Islam, and terrorism. To sum up, bias is an active area of research, and there's a lot of work to be done. I've seen examples of bias against women, gay people, and different races. **You don't want to be putting GPT-2 and BERT into production with text generation and masked modeling without some strict boundaries and having a human in the loop to check the output that is created.**
- BERT was trained on the English Wikipedia, which has around two and a half billion words, and something known as the BookCorpus, which is around 800 million words. The BooksCorpus are 11,000 books written by yet unpublished authors. GPT-2 was trained on WebText Corpus.
**Transformers Models with Tasks**
  Transformers are made up of two components: They are Encoder and decoder 
     1. BERT,ALBERT,Roberta and DistilBERT (Encoder only models)
           - Masked Language Modelling
           - Next Sentence Prediction
           - Text classification
           - named entity recognition
           - Question and answering
      2. BART(Facebook's Bidirectional and Auto Regressive Tranformers) or Google T5 model (Encoder and decoder models)
           - Translation
           - Text Summarization
      3. Decoder-only models are good for generative tasks, such as text generation. Examples include the GPT family such as GPT, GPT-2, and GPT-3.
          - Summarization
          - AI generated Fairytale
        
**Transformers**
The underlying architecture of BERT,GPT-3 and Different Large Language Models is Transformers. "Attention is all you need" by Google reasearchers introduced Transformers which became a turning point in Natural Language Processing(NLP).
