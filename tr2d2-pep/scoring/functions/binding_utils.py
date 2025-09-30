from torch import nn
import torch
import numpy as np

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
    
class MultiHeadAttentionSequence(nn.Module):
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_head*d_k)
        self.W_K = nn.Linear(d_model, n_head*d_k)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()

        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)

        attention = torch.matmul(Q, K)

        attention = attention / np.sqrt(self.d_k)

        attention = F.softmax(attention, dim=-1)
        
        output = torch.matmul(attention, V)

        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])
            
        output = self.W_O(output)

        output = self.dropout(output)
        
        output = self.layer_norm(output + q)
        
        return output, attention
        
class MultiHeadAttentionReciprocal(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
    
        self.W_Q = nn.Linear(d_model, n_head*d_k)
        self.W_K = nn.Linear(d_model, n_head*d_k)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)
        self.W_V_2 = nn.Linear(d_model, n_head*d_v)
        self.W_O_2 = nn.Linear(n_head*d_v, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, q, k, v, v_2):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()
        batch, len_v_2, _ = v_2.size()        
            
        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])
        V_2 = self.W_V_2(v_2).view([batch, len_v_2, self.n_head, self.d_v])
           
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)
        V_2 = V_2.transpose(1,2) 
        
        attention = torch.matmul(Q, K)
       
            
        attention = attention /np.sqrt(self.d_k)
        
        attention_2 = attention.transpose(-2, -1)
        
        
       
        attention = F.softmax(attention, dim=-1)
        
        attention_2 = F.softmax(attention_2, dim=-1)
    
        
        output = torch.matmul(attention, V)
        
        output_2 = torch.matmul(attention_2, V_2)
            
        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])
       
        output_2 = output_2.transpose(1, 2).reshape([batch, len_k, self.d_v*self.n_head])
            
        output = self.W_O(output)
        
        output_2 = self.W_O_2(output_2)
        
        output = self.dropout(output)
        
        output = self.layer_norm(output + q)
        
        output_2 = self.dropout(output_2)
        
        output_2 = self.layer_norm(output_2 + k)
       
        
        return output, output_2, attention, attention_2
        
    
class FFN(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        
        self.layer_1 = nn.Conv1d(d_in, d_hid,1)
        self.layer_2 = nn.Conv1d(d_hid, d_in,1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_in)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        residual = x 
        output = self.layer_1(x.transpose(1, 2))
        
        output = self.relu(output)
        
        output = self.layer_2(output)
        
        output = self.dropout(output)
        
        output = self.layer_norm(output.transpose(1, 2)+residual)
        
        return output

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class DilatedCNN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(DilatedCNN, self).__init__()
        self.first_ = nn.ModuleList()
        self.second_ = nn.ModuleList()
        self.third_ = nn.ModuleList()

        dilation_tuple = (1, 2, 3)
        dim_in_tuple = (d_model, d_hidden, d_hidden)
        dim_out_tuple = (d_hidden, d_hidden, d_hidden)

        for i, dilation_rate in enumerate(dilation_tuple):
            self.first_.append(ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=3, padding=dilation_rate,
                                         dilation=dilation_rate))

        for i, dilation_rate in enumerate(dilation_tuple):
            self.second_.append(ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=5, padding=2*dilation_rate,
                                          dilation=dilation_rate))

        for i, dilation_rate in enumerate(dilation_tuple):
            self.third_.append(ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=7, padding=3*dilation_rate,
                                         dilation=dilation_rate))

    def forward(self, protein_seq_enc):
        # pdb.set_trace()
        protein_seq_enc = protein_seq_enc.transpose(1, 2)    # protein_seq_enc's shape: B*L*d_model -> B*d_model*L

        first_embedding = protein_seq_enc
        second_embedding = protein_seq_enc
        third_embedding = protein_seq_enc

        for i in range(len(self.first_)):
            first_embedding = self.first_[i](first_embedding)

        for i in range(len(self.second_)):
            second_embedding = self.second_[i](second_embedding)

        for i in range(len(self.third_)):
            third_embedding = self.third_[i](third_embedding)

        # pdb.set_trace()

        protein_seq_enc = first_embedding + second_embedding + third_embedding

        return protein_seq_enc.transpose(1, 2)


class ReciprocalLayerwithCNN(nn.Module):

    def __init__(self, d_model, d_inner, d_hidden, n_head, d_k, d_v):
        super().__init__()

        self.cnn = DilatedCNN(d_model, d_hidden)

        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_hidden, d_k, d_v)

        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_hidden, d_k, d_v)

        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_hidden, d_k, d_v)

        self.ffn_seq = FFN(d_hidden, d_inner)

        self.ffn_protein = FFN(d_hidden, d_inner)

    def forward(self, sequence_enc, protein_seq_enc):
        # pdb.set_trace()  # protein_seq_enc.shape = B * L * d_model
        protein_seq_enc = self.cnn(protein_seq_enc)
        prot_enc, prot_attention = self.protein_attention_layer(protein_seq_enc, protein_seq_enc, protein_seq_enc)

        seq_enc, sequence_attention = self.sequence_attention_layer(sequence_enc, sequence_enc, sequence_enc)

        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(prot_enc, seq_enc, seq_enc, prot_enc)
        
        prot_enc = self.ffn_protein(prot_enc)

        seq_enc = self.ffn_seq(seq_enc)

        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention


class ReciprocalLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        
        super().__init__()
        
        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_model, d_k, d_v)
        
        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_model, d_k, d_v)
        
        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model, d_k, d_v)
        
        self.ffn_seq = FFN(d_model, d_inner)
        
        self.ffn_protein = FFN(d_model, d_inner)

    def forward(self, sequence_enc, protein_seq_enc):
        prot_enc, prot_attention = self.protein_attention_layer(protein_seq_enc, protein_seq_enc, protein_seq_enc)
        
        seq_enc, sequence_attention = self.sequence_attention_layer(sequence_enc, sequence_enc, sequence_enc)
        
        
        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(prot_enc, seq_enc, seq_enc, prot_enc)
        prot_enc = self.ffn_protein(prot_enc)
        
        seq_enc = self.ffn_seq(seq_enc)
        
        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention