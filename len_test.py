from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("asafaya/bert-base-arabic")

text = "بعد شهر من تدشين منظمة غاز شرق المتوسط.. قمة ثلاثية للمرة الثامنة في نيقوسيا.. زعماء مصر وقبرص واليونان يجتمعون خلال أيام لمناقشة الأوضاع في ليبيا وسوريا والاستفزازت التركية والتعاون في مجال الطاقة"
encode = tokenizer(text,return_tensors='pt')
print(len(encode['input_ids'][0]))
