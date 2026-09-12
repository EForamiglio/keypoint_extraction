# keypoint_extraction

Aplicação standalone para capturar keypoints de pose humana (YOLO pose) e
organizá-los em cenários de teste rotulados, para gerar dados de
validação/retreino do classificador GRU de detecção de quedas do PFC.

Só captura — não roda o classificador nem gera alertas. O rótulo do cenário é
escolhido antes de gravar, então nunca precisa ser reconstruído depois.

> Documentação completa de instalação e uso será adicionada ao final da
> implementação (ver plano do projeto).
