from time import time

import torch
from tqdm import tqdm

def test_model(model, test_loader, device, metrics, batch_size):
    model.to(device)

    # modelo em modo de validacao
    model.eval()

    iterator = tqdm(
        enumerate(test_loader),
        desc="test_loop",
        unit="batches",
    )

    total_time = time()
    with torch.inference_mode():
        for i, (labels, imgs) in iterator:
            batch_time = time()
            labels = labels.to(device)
            imgs = imgs.to(device)

            # forward pass
            outputs = model(imgs)

            predicted_class = torch.max(torch.log_softmax(outputs, dim=1), dim=1)[1]

            print(
                f"Time to execute batch {i} of size {len(labels)}: {batch_time - time()}s\n\nMétricas da batch {metrics(predicted_class, labels) = }"
            )

        print(
            f"Tempo de execução sobre a base de testes: {total_time - time()}s\n\n Métricas sobre os dados de teste: {metrics.compute() = }"
        )
        
        
def main():
    
    MODEL_PATH = "/Users/luishf/Documents/GitHub/resnet-finetuning/models/05:13-23:50.pt"
    DEVICE = torch.device("mps")
    BATCH_SIZE = 1
    
    model = torch.load(MODEL_PATH)
    
    #TODO inserir aqui o loader para os dados de teste
    test_loader = ...
    
    
    test_model(model=model, test_loader=test_loader, device=DEVICE, batch_size=BATCH_SIZE)
        
        
if __name__ == "__main__":
    main()
