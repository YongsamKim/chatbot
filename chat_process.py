from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import random

class emochatbot():
  def __init__(self):
    bos_token = '<s>'
    eos_token = '</s>'
    pad_token = '<pad>'
    #kogpt2 모델 load"
    self.Tokenizer_kogpt2 = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                                    bos_token = bos_token, 
                                                                    eos_token = eos_token,
                                                                    pad_token = pad_token)
    self.model_kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    #kcelectra 모델 load
    self.Tokenizer_kce = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    self.model_kce = AutoModel.from_pretrained("beomi/KcELECTRA-base", num_labels=6)
    # finetuning된 모델 load
    checkpoint_kogpt = torch.load('./models/emo_chat_model.pth',
                                  map_location=torch.device('cpu'))
    self.model_kogpt2.load_state_dict(checkpoint_kogpt['model_state_dict'])
    checkpoint_kce = torch.load('./models/emo_classify_model.pt',
                                  map_location=torch.device('cpu'))
    self.model_kce.load_state_dict(checkpoint_kce)

  def sent_gen(self,sent):
    self.model_kogpt2.eval()
    sent = str(sent)
    tokenized_sent = self.Tokenizer_kogpt2.encode(sent)
    input_ids = torch.tensor([self.Tokenizer_kogpt2.bos_token_id] + tokenized_sent + [self.Tokenizer_kogpt2.eos_token_id]).unsqueeze(0)
    output_token =  self.model_kogpt2.generate(input_ids = input_ids,
                                             do_sample = True,
                                             max_length = 70,
                                             min_length = 35,
                                             top_p=0.92,
                                             top_k = 50,
                                             temperature = 0.6,
                                             no_repeat_ngram_size = None,
                                             num_return_sequences=3,
                                             early_stopping=False)
    output_gen = self.Tokenizer_kogpt2.decode(output_token[0].tolist()[len(tokenized_sent)+1:], skip_special_tokens = True)
    return output_gen

  def clssify_emo(self, sent):
    self.model_kce.eval()
    inputs =  self.Tokenizer_kce(sent,
                                 return_tensors='pt',
                                 truncation = True,
                                 max_length=64,
                                 pad_to_max_length = True,
                                 add_special_tokens = True)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    out = self.model_kce(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
    out = out[:, -1, :]
    # 0 : happiness, 1 : angry,  2 : sadness, 3 : disgust, 4 : surprise, 5 : fear
    emotion = np.argmax(out.detach().cpu().numpy())
    return emotion

  def recommendation(self, sent):
    answer = self.sent_gen(sent)
    emotion_status = self.clssify_emo(sent)
    
    if emotion_status == 0:
      status = 'happiness'
    elif emotion_status == 1:
      status = 'angry'
    elif emotion_status == 2:
      status = 'sadness'
    elif emotion_status == 3:
      status = 'disgust'
    elif emotion_status == 4:
      status = 'surprise'
    elif emotion_status == 5:
      status = 'fear'
    emoclass = f"오늘의 기분은 {status}로 판단됩니다."

    if emotion_status == 0:
      mood = '행복'
    elif emotion_status == 1:
      mood = '화남'
    elif emotion_status == 2:
      mood = '슬픔'
    elif emotion_status == 3:
      mood = '차분함'
    elif emotion_status == 4:
      mood = '극적'
    elif emotion_status == 5:
      mood = '어두움'
    #현재 경로는 임시경로
    #music_list = os.listdir('https://ssacteam2.s3.us-west-1.amazonaws.com/' + mood)
    #rand_music_reco = random.sample(music_list, 5)
    rand_music_reco = "https://ssacteam2.s3.us-west-1.amazonaws.com/Sad/80's+Video+Game+Death+-+Sir+Cubworth.mp3"

    return answer, emoclass, rand_music_reco

#이하 실행 테스트
#test = emochatbot()

#answer, emoclass, music_list = test.recommendation('일을 자기가 안 하고 나한테 해달라고 해서 번거로워')
#print(answer)
#print(emoclass)
#print(music_list)