
from argparse import ArgumentParser
from SASREC.A_SASRec_final_bce_llm import SASRec
from SASREC.SASRecModules_ori import *
# from A_SASRec_final_bce_llm import SASRec
# from SASRecModules_ori import *
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import pickle
import random
import csv
import os



os.environ["OPENAI_API_KEY"] = "sk-ACmwjXam65oCkcAQ66AfEcD894984f91BbEbE162765d502e"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"


def generate_item_embedding(item_text_dic, tokenizer, model, word_drop_ratio=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order_texts = [[0]] * (len(item_text_dic))   # 列表，列表中的第i个元素代表ID为i的项目对应的文本。创建一个项目长度的列表

    # for item in item_text_dic:
    #     order_texts[item] = item_text_dic[item]# 变成嵌套列表的形式[[1,2,3],[4,5,6]],数字部分为内容

    # # 生成embedding时
    max_item_id = max(item_text_dic.keys())
    order_texts = ["" if k == 0 else item_text_dic.get(k, "") for k in range(max_item_id + 1)]
    #for text in order_texts:
    #    assert text != [0]


    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]# 一次处理batch_size的内容
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                '''
                split 的意思是：将一整个完整的字符串分隔为各个单词，这个单词按照顺序构成一个列表
                例如：
                sent = "Hello world this is a test" 
                sent = sent.split(' ')，   那么输出是：
                ['Hello', 'world', 'this', 'is', 'a', 'test']
                '''
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(device)
        outputs = model(**encoded_sentences) #编译后的字段放入模型中进行推理
        # if args.emb_type == 'CLS':
        #     cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()    # 取第一个时刻的表示？？
        #     embeddings.append(cls_output)
        # elif args.emb_type == 'Mean':

        # output隐藏层：[batch_size, seq_len, hidden_size]
        # 计算平均池化嵌入
        masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1) # [batch_size, seq_len, 1]
        mean_output = masked_output[:,1:,:].sum(dim=1) / encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
        mean_output = mean_output.detach()
        embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    # 猜测embeddings的大小为：项目总数 * 嵌入大小

    print('Embeddings shape: ', embeddings.shape)
    return embeddings



def load_plm(model_name='bert-base-uncased'):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, quantization_config=bnb_config)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()

    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--precision', default='bf16', type=str)
    parser.add_argument('--amp_backend', default="native", type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--accumulate_grad_batches', default=8, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)

    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)

    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    parser.add_argument('--dataset', default='movielens_data', type=str)
    parser.add_argument('--data_dir', default='data/ref/movielens1m', type=str)
    parser.add_argument('--model_name', default='mlp_projector', type=str)
    parser.add_argument('--loss', default='lm', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    parser.add_argument('--log_dir', default='movielens_logs', type=str)
    
    parser.add_argument('--rec_size', default=64, type=int)
    parser.add_argument('--padding_item_id', default=1682, type=int)
    parser.add_argument('--llm_path', type=str)
    parser.add_argument('--rec_model_path', default='./rec_model/SASRec_ml1m.pt', type=str)
    parser.add_argument('--prompt_path', default='./prompt/movie/', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--rec_embed', default="SASRec", choices=['SASRec', 'Caser','GRU'], type=str)

    parser.add_argument('--aug_prob', default=0.5, type=float)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--metric', default='hr', choices=['hr'], type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--save', default='part', choices=['part', 'all'], type=str)
    parser.add_argument('--cans_num', default=10, type=int)

    # Finetuning
    parser.add_argument('--llm_tuning', default='lora', choices=['lora', 'freeze','freeze_lora'], type=str)
    parser.add_argument('--peft_dir', default=None, type=str)
    parser.add_argument('--peft_config', default=None, type=str)
    parser.add_argument('--lora_r', default=8, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    args = parser.parse_args()
    
    if 'movielens' in args.data_dir:
        args.padding_item_id = 1682
    elif 'steam' in args.data_dir:
        args.padding_item_id = 3581
    elif 'lastfm' in args.data_dir:
        args.padding_item_id = 4606

    #main(args)

    # item mapping
    item_description = {}  # key：项目ID，值：项目描述
    user2idx = {}  # key：用户ID，值：用户索引
    item_text_dic = {}  # key：项目ID，值：项目索引
    # 读取数据集ai_user_summary_output_fixed
    user_i=0
    item_i=0
    user_interactions = dict()
    user_item_label = []
    user_item_label_test = []
    user_interactions_test = dict()
    user_profile = dict()
    with open('DRL-code-pytorch-main/Course_DQN/SASREC/ai_user_summary_output_fixed.csv', 'r', encoding='utf-8') as f:
    # with open('SASREC/ai_user_summary_output_fixed.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            user = row[0]
            
            profile = row[1]
            item1_text = row[2]
            item2_text = row[3]
            item3_text = row[4]
            item4_text = row[5]
            item5_text = row[6]
            user2idx[user]=user_i
            user_interactions[user_i] = []
            user_interactions_test[user_i] = []
            # item_text_dic是所有课程的字典，存内容{0:"xxx", 1:"xxxx",...}
            # user_interactions
            # item_id = len(item_text_dic) + 1  # 当前项目的ID，从1开始，留0给padding
            # item_text_dic[item_id]=item1_text
            # user_interactions[user_i].append(item_id)
            # user_interactions_test[user_i].append(item_id)
            # 替换原有的 item_text_dic 和 user_interactions 构造
            for item_text in [item1_text, item2_text, item3_text]:
                item_id = len(item_text_dic) + 1
                item_text_dic[item_id] = item_text
                user_interactions[user_i].append(item_id)
                user_interactions_test[user_i].append(item_id)

            item_id = len(item_text_dic) + 1
            item_text_dic[item_id]=item4_text
            #user_interactions[user_i].append(len(item_text_dic))
            user_interactions_test[user_i].append(item_id)
            user_item_label.append(item_id)

            item_id = len(item_text_dic) + 1
            item_text_dic[item_id]=item5_text
            user_interactions_test[user_i].append(item_id)
            #user_interactions[user_i].append(len(item_text_dic))
            user_item_label_test.append(item_id)

            user_profile[user_i] = profile

            user_i+=1
    # print(item_text_dic)
    # print(item_text_dic.keys())
    # print(user_interactions)
    # print(user_interactions_test)
    # print(user_item_label_test)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plm_tokenizer, plm_model = load_plm()
    plm_model = plm_model.to(device)

    item_text_embeddings = generate_item_embedding(item_text_dic, plm_tokenizer, plm_model, word_drop_ratio=-1)    # 类型是torch张量,总的一个
    # print(item_text_embeddings)  # 输出形状，应该是 (项目总数, 嵌入维度)
    # 把这个序列化写成文件，保存中间结果
    with open('DRL-code-pytorch-main/Course_DQN/SASREC/item_text_embeddings.pickle', 'wb') as f:
        pickle.dump(item_text_embeddings, f)

    with open('DRL-code-pytorch-main/Course_DQN/SASREC/item_text_embeddings.pickle', 'rb') as f:
        item_text_embeddings = pickle.load(f)

    
    user_items = []
    for user in user_interactions:
        user_items.append(torch.tensor(user_interactions[user]))  # 将列表转换为Tensor
    user_items = torch.stack(user_items, dim=0)  # 用户总数 * 5


    user_items_test = []
    for user in user_interactions_test:
        user_items_test.append(torch.tensor(user_interactions_test[user]))
    user_items_test = torch.stack(user_items_test, dim=0)  # 用户总数 * 5
    # print(user_items_test)


    # 将item_text_embeddings转化为张量
    item_text_embeddings = torch.tensor(item_text_embeddings, dtype=torch.float32).to(device)

    user_item_label = torch.tensor(user_item_label, dtype=torch.long).to(device)

    user_item_label_test = torch.tensor(user_item_label_test, dtype=torch.long).to(device)
    # print(item_text_embeddings)
    # print(item_text_embeddings.size())
    # print(user_items)
    # print(user_items.size())
    # print(user_items_test)
    # print(user_items_test.size())

    # 初始化SASRec模型
    SASRec = SASRec(
        hidden_size=768,
        item_num=item_text_embeddings.shape[0],
        state_size=768,
        dropout=0.1,
        device=device,
        num_heads=1,
        pre_embeddings=item_text_embeddings  # 传入张量形式的项目嵌入
    )
    print("item_text_emb:", item_text_embeddings.shape[0])
    SASRec.to(device)
    # Ensure len_states matches the sequence length of user_items
    sequence_length = user_items.size(1)  # Get the sequence length from user_items
    len_states = torch.tensor([sequence_length] * user_items.size(0)).to(device)
    # print(len_states)
    
    sequence_length_test = user_items_test.size(1)  # Get the sequence length from user_items_test
    len_states_test = torch.tensor([sequence_length_test] * user_items_test.size(0)).to(device)
    # print(len_states_test)
    print("user_items.shape:", user_items.size(), "len_states:", len_states)
    # user_items = user_items.to(device) 
    user_items = user_items.to(device)
    len_states = len_states.to(device)
    # 训练模型
    optimizer = torch.optim.Adam(SASRec.parameters(), lr=0.001)
    print("user_items_test max:", user_items_test.max().item(), "min:", user_items_test.min().item())
    print("item_num:", item_text_embeddings.shape[0])
    SASRec.train()  # 设置模型为训练模式
    for epoch in range(50):
        output = SASRec(user_items, len_states)    # 输出每一个用户，对所有候选项目的预测得分    用户总数 * 候选项目数

        print(output.size())

        # 计算损失，即交叉熵损失
        loss = torch.nn.functional.cross_entropy(output, user_item_label)
        print(loss)  # 输出损失值
        
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()

    # 测试模型
    SASRec.eval()  # 设置模型为评估模式
    with torch.no_grad():
        test_output = SASRec(user_items_test, len_states_test)  # 输出每一个用户，对所有候选项目的预测得分

        # 找出预测得分最高的前10个项目
        _, predicted_indices = torch.topk(test_output, k=10, dim=1)
        print(predicted_indices)


    # 计算召回率
    hits = 0
    for i in range(len(user_item_label_test)):
        if user_item_label_test[i] in predicted_indices[i]:
            hits += 1
    recall = hits / len(user_item_label_test)
    print(f"Recall: {recall:.4f}")


    # 保存整个模型架构
    torch.save(SASRec.state_dict(), 'DRL-code-pytorch-main/Course_DQN/SASREC/SASRec_model.pth')
