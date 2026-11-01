"""Float64 full-vocabulary KL with parallel reductions over vocabulary blocks."""


def reduce_chunk(torch,a,b,block=4096):
    if a.ndim!=2 or a.shape!=b.shape or a.shape[-1]%block:
        raise ValueError('Invalid blocked vocabulary reduction shape')
    aa=a.reshape(a.shape[0],-1,block)
    bb=b.reshape(b.shape[0],-1,block)
    za=torch.logsumexp(torch.logsumexp(aa,dim=-1),dim=-1)
    zb=torch.logsumexp(torch.logsumexp(bb,dim=-1),dim=-1)
    pa=a-za[:,None]
    pb=b-zb[:,None]
    terms=(pa.exp()*(pa-pb)).reshape(a.shape[0],-1,block)
    kl=terms.sum(dim=-1).sum(dim=-1)
    def argmax(values):
        maxima,indices=values.max(dim=-1)
        chosen=maxima.argmax(dim=-1,keepdim=True)
        return chosen[:,0]*block+indices.gather(-1,chosen)[:,0]
    agree=(argmax(aa)==argmax(bb)).sum()
    return kl,agree
