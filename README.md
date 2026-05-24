# Deep Reinforcement Learning for Portfolio Optimization

Repositorio correspondiente al Trabajo Final de Máster centrado en la aplicación de técnicas de Deep Reinforcement Learning (DRL) para problemas de optimización dinámica de carteras financieras.

El proyecto desarrolla un entorno financiero basado en Gymnasium y un agente PPO (*Proximal Policy Optimization*) implementado en PyTorch, con el objetivo de aprender políticas de asignación de activos sobre una cartera de ETFs.

## Características principales

- Implementación de un entorno financiero personalizado basado en Gymnasium.
- Agente PPO basado en arquitectura Actoro-Critico.
- Optimización dinámica de carteras mediante Deep Reinforcement Learning.
- Reward basada en Sharpe Ratio rolling y penalización de turnover.
- Evaluación comparativa frente a estrategias clásicas de inversión.
- Visualización de métricas de entrenamiento y evolución de pesos de cartera.

## Estructura del repositorio

```text
.
├── agent.py
├── portfolioenv.py
├── train.ipynb
├── outputs/
├── requirements.txt
└── README.md
```

### Descripción de archivos

- `agent.py`: implementación del agente PPO y proceso de entrenamiento.
- `portfolioenv.py`: entorno financiero utilizado durante el entrenamiento y evaluación.
- `train.ipynb`: notebook principal con entrenamiento, evaluación y generación de resultados.
- `outputs/`: gráficas y resultados obtenidos durante los experimentos.


## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el notebook principal:

```bash
jupyter notebook train.ipynb
```

## Autor

Nikolas Roteta
