"""Unchanged passive Observer, extracted without private cluster authorization."""
import copy,hashlib,json,time
from pathlib import Path
import torch

class Observer:
    """CPU-offset-free token digest and scalar ledger; changes no model update.

    Stores only CPU input bytes/coefficients after use. Restore digest and records
    from the checkpoint, never infer replay history from the current output path.
    """
    def __init__(self,out,c,model,claim=None,old=None):
        self.out=Path(out);self.c=c;self.claim=claim
        self.chain='0'*64;self.microbatches=0;self.updates=[]
        if old:
            self.chain=old['chain'];self.microbatches=old['microbatches'];self.updates=copy.deepcopy(old['updates'])
        self.parameters=sum(p.numel() for p in model.parameters())
        self.started=time.monotonic()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        self.pending_signature=self.chain
        self.validation_signature=None

    def sample(self,x,y):
        # The existing data iterator returns CPU int64 tensors in a fixed shape.
        h=hashlib.sha256(bytes.fromhex(self.chain))
        for token in (x,y):
            assert token.device.type=='cpu' and token.dtype==torch.int64
            h.update(token.contiguous().numpy().tobytes())
        self.chain=h.hexdigest();self.microbatches+=1

    def after_update(self,model,update,seconds,lr):
        model.check_coefficients()
        scalars=[]
        if self.c.method!='B':
            for block in model.net.blocks:
                scalars.extend((float(block.beta1().detach()),float(block.beta2().detach())))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite executed model parameter')
        row=dict(update=update,tokens=update*self.c.tokens_per_update,lr=lr,
                 training_microbatches=self.microbatches,sample_chain=self.chain,
                 beta_min=min(scalars) if scalars else None,beta_max=max(scalars) if scalars else None,
                 scalar_count=len(scalars),optimizer_seconds=seconds)
        self.updates.append(row)
        with (self.out/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def validation_identity(self,iterator,c):
        state=copy.deepcopy(iterator.state_dict());h=hashlib.sha256()
        try:
            for _ in range(c.eval_batches):
                x,y=next(iterator)
                for token in (x,y):h.update(token.contiguous().numpy().tobytes())
        finally:iterator.load_state_dict(state)
        self.validation_signature=h.hexdigest()
        return self.validation_signature

    def state_dict(self):
        return dict(chain=self.chain,microbatches=self.microbatches,updates=self.updates)

    def manifest(self,protocol,data_identity,model):
        return dict(schema='e063_quality_v1',protocol=protocol,data_identity=data_identity,
                    release=self.claim,validation_signature=self.validation_signature,
                    parameters=self.parameters,unique_blocks=len(model.net.blocks),logical_blocks=8,
                    precision='IEEE FP32;TF32 off',quality_run=self.claim is not None,
                    fixture_cpu_only=self.claim is None and not next(model.parameters()).is_cuda)

    def finish(self,result,model,completed,inference):
        result.update(quality_run=self.claim is not None,
                      fixture_cpu_only=self.claim is None and not next(model.parameters()).is_cuda,
                      executed_beta_gate='0<beta<1 every update',sample_chain=self.chain,
                      validation_signature=self.validation_signature,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if next(model.parameters()).is_cuda else None,
                      inference_rows=len(inference),updates_recorded=len(self.updates),
                      tokens_per_second=result['tokens']/max(result['elapsed_seconds'],1e-30),
                      source_lock_sha256=self.claim['source_lock_sha256'] if self.claim else None,
                      approval_sha256=self.claim['approval_sha256'] if self.claim else None)
        # Restored history is emitted completely, not just the resumed suffix.
        with (self.out/'updates.jsonl').open('w') as f:
            for row in self.updates:f.write(json.dumps(row,allow_nan=False)+'\n')
        return result
