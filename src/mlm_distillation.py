import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

from util.parse_json import parse_json
from util.saving import model_saving, loss_saving

class MLMDistill():
    def __init__(self, config) -> None:
        self.config = config
        self.model = SentenceBertLM(
            bert=BertModel.from_pretrained(config.model)
        )
        self.tokenizer = load_tokenizer(config.model)
         
    def prepare_train_set(self):
        corpus = get_recog_data(self.config.train_data_path, self.config.text_type, self.config.max_utts)
        labels = get_recog_data(self.config.train_data_path, "hyp_score", self.config.max_utts)

        # 過濾過長的seq，並把每個seq複製和它長度一樣的次數
        ids_list = []
        label_list = []
        attention_mask_list = []

        for utt_hyps_text, utt_hyps_score in zip(corpus, labels):
            for hyp_text, hyp_score in zip(utt_hyps_text, utt_hyps_score):

                token_seq = self.tokenizer.tokenize(hyp_text)
                if len(token_seq) < self.config.max_seq_len:
                    ids_list.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + token_seq + ["[SEP]"]
                        )
                    )

                    label_list.append(hyp_score)

                    attention_mask_list.append([1]*len(["[CLS]"] + token_seq + ["[SEP]"]))
        
        self.train_set = self.trainDataset(ids_list, label_list, attention_mask_list)
        
        return self.train_set

    def prepare_train_loader(self):
        self.train_loader = DataLoader(
            dataset=self.train_set,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
        )
        return self.train_loader

    def collate(self, data):
        id_tensor, label, mask_tensor = zip(*data)
        
        id_tensor = pad_sequence(id_tensor, batch_first=True)
        mask_tensor = pad_sequence(mask_tensor, batch_first=True)
        label_tensor = torch.tensor(label)
        return id_tensor, label_tensor, mask_tensor

    def train(self):
        train_loss_record = [0]*self.config.epoch
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch_id in range(1, self.config.epoch+1):
            print("Epoch {}/{}".format(epoch_id, self.config.epoch))

            self.model = self.model.to(self.config.device)
            self.model.train()
            
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for _, (id_tensor, label_tensor, mask_tensor) in loop:

                id_tensor = id_tensor.to(self.config.device)
                label_tensor = label_tensor.to(self.config.device)
                mask_tensor = mask_tensor.to(self.config.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    output = self.model(
                        input_ids=id_tensor,
                        attention_mask=mask_tensor
                    )
                    output = torch.squeeze(output)
                    loss = MSELoss()(output, label_tensor)
                    loss.backward()
                    optimizer.step()
                
                train_loss_record[epoch_id-1] += loss.item()
            
            train_loss_record[epoch_id-1] /= len(self.train_loader)
            print("epoch ", epoch_id, "loss: ", train_loss_record[epoch_id-1])

            torch.save(
                self.model.state_dict(),
                os.path.join(self.config.output_path, 'checkpoint_{}.pth'.format(epoch_id))
            )

            with open(self.config.output_path + "loss.txt", "w") as f:
                f.write("epoch loss: \n")
                f.write(" ".join([str(loss) for loss in train_loss_record]))

    class trainDataset(Dataset):
        def __init__(self, ids_list, label_list, attention_mask_list):
            self.ids_list = ids_list
            self.label_list = label_list
            self.attention_mask_list = attention_mask_list

        def __len__(self):
            return len(self.ids_list)
        
        def __getitem__(self, idx):
            ids_tensor = torch.tensor(self.ids_list[idx], dtype=torch.long)
            label = self.label_list[idx]
            attention_mask_tensor = torch.tensor(self.attention_mask_list[idx], dtype=torch.long)
            return ids_tensor, label, attention_mask_tensor