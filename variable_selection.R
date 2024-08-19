library(tidyverse)
library(ISLR2)
library(randomForest)
library(xgboost)

# importamos los datos
datos_pacientes <- read.csv("ReplicatedAcousticFeatures-ParkinsonDatabase.csv")
datos_pacientes <- datos_pacientes %>% as_tibble()

# seleccionamos las variables
datos_pacientes <- datos_pacientes %>% select(-one_of(c("ID", "Recording", "Gender")))

# convertimos a tipo factor nuestra variable objetivo
datos_pacientes$Status <- as.factor(datos_pacientes$Status)

###### RANDOM FOREST

# primer modelo 5000 arboles aleatorios
rf.model1 <- randomForest(Status ~ ., data = datos_pacientes,
                          importance = T, 
                          ntree=20000, 
                          keep.forest=FALSE)

# grafico auxiliar
# vemos las variables que nos podrían ser de ayuda al momento de predecir
varImpPlot(rf.model1)

# guardamos las variables y procederemos a filtrar
important_variables <- as_tibble(importance(rf.model1), rownames = "variable")

## codigo complementario

ggplot(important_variables, aes(x = reorder(variable, -MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(aes(fill = MeanDecreaseGini >= 2), stat = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "gray")) +
  geom_text(aes(label = round(MeanDecreaseGini, 1)), vjust = -0.3) +
  labs(x = "Variable",
       y = "Mean Decrease Gini",
       fill = "MDG >= 2") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(important_variables, aes(x = reorder(variable, -MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(aes(fill = MeanDecreaseAccuracy >= 30), stat = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "gray")) +
  geom_text(aes(label = round(MeanDecreaseAccuracy, 0)), vjust = -0.3) +
  labs(x = "Variable",
       y = "Mean Decrease Accuracy",
       fill = "MDA >= 30") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## fin de codigo complementario

# nos quedamos con las variables que tienen un mean decrease gini mayor/igual a 2
subset.dec.gini <- important_variables %>% 
  # filter(MeanDecreaseGini >=2) %>% 
  arrange(desc(MeanDecreaseGini)) %>% 
  select(c(variable, MeanDecreaseGini))

# nos quedamos con las variables que tienen un mean decrease acc mayor/igual a
subset.dec.acc <- important_variables %>% 
  filter(MeanDecreaseAccuracy >= 30) %>% 
  arrange(desc(MeanDecreaseAccuracy)) %>% 
  # select(variable)

###### BOOSTED TREES

# primer modelo ajustado
bt.model1 <- xgboost(data = as.matrix(datos_pacientes[,-1]), 
                     label = as.numeric(as.character(datos_pacientes$Status)),
                     nrounds = 1000,
                     objective = "binary:logistic")

# guardamos las metricas de cada variable para llevar a cabo la seleccion
important_variables <- xgb.importance(model = bt.model1)

# grafico auxiliar
# tenemos que las variables Delta0 y HNR38 son las que destacan
xgb.plot.importance(important_variables)

# graficamos las primeras variables, vemos que ayudann bastante a separar los puntos
ggplot(datos_pacientes, aes(x=Delta0, y=HNR38, color=Status)) +
  geom_point(size = 3, alpha = 0.8)

# nos quedamos con las variables tal que su importancia sea mayor a 0.01
important_variables_tibble <- important_variables %>% 
  as.tibble() %>% 
  head(25) %>%
  select(c(Feature, Gain))

ggplot(important_variables_tibble, 
       aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(aes(fill = Gain >= 0.01), stat = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "gray")) +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.3) +
  labs(x = "Variable",
       y = "Gain",
       fill = "Gain >= 0.01") +
  theme_minimal() +
  #theme(axis.text.y = element_text(angle = 45, hjust = 1)) +
  coord_flip()

subset.imp.xgb <- as.tibble(important_variables[1:25, 1])

############# subconjuntos finales

# utilizaremos solamente 4 subconjuntos:
# - los dos creados con random forest
# - el creado por xgboos
# - la interseccion de todos los subconjuntos anteriores

# interseccion de todos los subconjuntos
vars.rf <- semi_join(subset.dec.gini, subset.dec.acc)
vars <- semi_join(vars.rf, subset.imp.xgb, by = c("variable" = "Feature"))

# subconjuntos seleccionados vemos si hay diferencias
subset.dec.gini$variable %in% subset.dec.acc$variable
subset.dec.acc$variable %in% subset.dec.gini$variable
subset.imp.xgb$Feature %in% subset.dec.gini$variable
subset.imp.xgb$Feature %in% subset.dec.acc$variable

View(subset.dec.gini)
View(subset.dec.acc)
View(subset.imp.xgb)
View(vars)