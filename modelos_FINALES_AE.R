# ==============================================================================
#              PROYECTO FINAL - PREDICCIÓN DE DEPRESIÓN
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------------------------

library(caret)
library(pROC)
library(e1071)
library(naivebayes)
library(class)
library(klaR)
library(MASS)
library(ggplot2)
library(nnet)  # Para la red neuronal
library(rpart)         # Arbol de decision
library(ipred)         # Bagging
library(randomForest)  # Random Forest
library(adabag)        # ADA Boost
library(gbm)           # Gradient Boost
library(dplyr)


# ------------------------------------------------------------------------------
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# ------------------------------------------------------------------------------

# Lectura de datos
base <- read.csv("student_depression_dataset.csv")
dim(base)

# Eliminar columna id (primera columna)
df <- base[, -1]
str(df)

# Verificar valores faltantes en Financial.Stress
table(df$Financial.Stress, useNA = "ifany")

# Imputar Financial.Stress con mediana (si hay "?" o NA)
med_fs <- median(df$Financial.Stress, na.rm = TRUE)
for (i in 1:nrow(df)) {
  if(is.na(df$Financial.Stress[i]) || df$Financial.Stress[i] == "?") {
    df$Financial.Stress[i] <- med_fs
  }
}
df$Financial.Stress <- as.numeric(df$Financial.Stress)

# Transformar variables nominales a factor
cols_factor <- c(1, 3, 4, 12, 13, 16, 17)
print(colnames(df)[cols_factor])
df[cols_factor] <- lapply(df[cols_factor], as.factor)

# Transformar variables ordinales a factor ordenado
cols_factor_ord <- c(5, 6, 8, 9, 15)
print(colnames(df)[cols_factor_ord])
for (i in seq_along(cols_factor_ord)) {
  v <- cols_factor_ord[i]
  df[, v] <- factor(df[, v], ordered = TRUE)
}

# Reordenar niveles de Sleep.Duration y Dietary.Habits
df[, 10] <- factor(df[, 10], 
                   levels = c("Others", "'Less than 5 hours'", "'5-6 hours'", 
                              "'7-8 hours'", "'More than 8 hours'"),
                   ordered = TRUE)

df[, 11] <- factor(df[, 11], 
                   levels = c("Others", "Unhealthy", "Moderate", "Healthy"),
                   ordered = TRUE)

# Verificar variables de baja variabilidad
print(table(df$Work.Pressure, useNA = "ifany"))
print(table(df$Job.Satisfaction, useNA = "ifany"))
print(table(df$Profession, useNA = "ifany"))

# Eliminar variables con baja variabilidad (usando sintaxis base R)
cols_eliminar <- c("Work.Pressure", "Job.Satisfaction", "Profession")
df <- df[, !(names(df) %in% cols_eliminar)]

#Dimensiones después de eliminar variables:
dim(df)
#Variables finales:
print(names(df))

# ------------------------------------------------------------------------------
# 3. PARTICIÓN TRAIN/TEST (70/30)
# ------------------------------------------------------------------------------
set.seed(1975)
ntrain <- round(0.7 * nrow(df))
auxtrain <- sample(nrow(df), ntrain)
df_train <- df[auxtrain, ]
df_test <- df[-auxtrain, ]

#Train:
nrow(df_train)

#Test:
nrow(df_test)

#Distribución Depression en Train:
print(prop.table(table(df_train$Depression)))

# ------------------------------------------------------------------------------
# 4. VALIDACIÓN CRUZADA MANUAL (5-Fold)
# ------------------------------------------------------------------------------
set.seed(1234)
K <- 5
n <- nrow(df_train)
idx <- sample(1:n, n, replace = FALSE)
folds <- split(idx, cut(seq_along(idx), K, labels = FALSE))

# Validación cruzada:
K

# ==============================================================================
#                           5. NAIVE BAYES--------------------------------------
# ==============================================================================

laplace_grid <- c(0, 0.5, 1, 2)
resultados_nb <- data.frame(laplace = numeric(), AUC = numeric(), SD = numeric())

for(lap in laplace_grid) {
  
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_fold <- df_train[train_idx, ]
    test_fold <- df_train[test_idx, ]
    
    # Modelo Naive Bayes
    modelo_nb <- naive_bayes(Depression ~ ., data = train_fold, laplace = lap)
    
    # Predicción de probabilidades
    prob_nb <- predict(modelo_nb, test_fold, type = "prob")[, "1"]
    
    # Cálculo de AUC
    roc_nb <- roc(test_fold$Depression, prob_nb, levels = c("0", "1"), 
                  direction = "<", quiet = TRUE)
    auc_vals[i] <- auc(roc_nb)
  }
  
  resultados_nb <- rbind(resultados_nb, data.frame(
    laplace = lap, AUC = mean(auc_vals), SD = sd(auc_vals)
  ))
  cat("Laplace =", lap, "| AUC =", round(mean(auc_vals), 4), 
      "±", round(sd(auc_vals), 4), "\n")
}

# Mejor configuración
mejor_nb <- resultados_nb[which.max(resultados_nb$AUC), ]
cat("\n>>> Mejor Naive Bayes: laplace =", mejor_nb$laplace, 
    "| AUC =", round(mejor_nb$AUC, 4), "\n")

saveRDS(mejor_nb, "Mejor Naive Bayes")
mejor_nb

# ==============================================================================
#                              6. KNN-------------------------------------------
# ==============================================================================

# Preparar matriz de predictores numéricos
formula_knn <- Depression ~ .
X_train <- model.matrix(formula_knn, data = df_train)[, -1]
y_train <- df_train$Depression
X_test <- model.matrix(formula_knn, data = df_test)[, -1]
y_test <- df_test$Depression

# Normalización con manejo de varianza cero
X_train_scaled <- scale(X_train)
center_vals <- attr(X_train_scaled, "scaled:center")
scale_vals <- attr(X_train_scaled, "scaled:scale")

# Identificar columnas con varianza cero (scale = 0 o NA)
cols_var_cero <- which(is.na(scale_vals) | scale_vals == 0)
if(length(cols_var_cero) > 0) {
  # Eliminando variables con varianza cero
  length(cols_var_cero)
  print(names(cols_var_cero))
  
  # Eliminar esas columnas
  X_train <- X_train[, -cols_var_cero]
  X_test <- X_test[, -cols_var_cero]
  
  # Volver a normalizar sin esas columnas
  X_train_scaled <- scale(X_train)
  center_vals <- attr(X_train_scaled, "scaled:center")
  scale_vals <- attr(X_train_scaled, "scaled:scale")
}

# Aplicar normalización al test
X_test_scaled <- scale(X_test, center = center_vals, scale = scale_vals)

# Reemplazar cualquier NA o NaN restante por 0
X_train_scaled[is.na(X_train_scaled)] <- 0
X_test_scaled[is.na(X_test_scaled)] <- 0

# Variables para KNN:
ncol(X_train_scaled)
# Verificando NA en X_train_scaled:
sum(is.na(X_train_scaled))
# Verificando NA en X_test_scaled:
sum(is.na(X_test_scaled))

# Grid de K
k_grid <- c(1, 5, 11, 21, 51, 101, 151, 201, 251, 301, 401, 480)
resultados_knn <- data.frame(k = numeric(), AUC = numeric(), SD = numeric())

for(k_val in k_grid) {
  
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    X_train_fold <- X_train_scaled[train_idx, ]
    X_test_fold <- X_train_scaled[test_idx, ]
    y_train_fold <- y_train[train_idx]
    y_test_fold <- y_train[test_idx]
    
    # Modelo KNN
    pred_knn <- knn(train = X_train_fold, test = X_test_fold, 
                    cl = y_train_fold, k = k_val, prob = TRUE)
    
    # Obtener probabilidades
    prob_votos <- attr(pred_knn, "prob")
    prob_class1 <- ifelse(pred_knn == "1", prob_votos, 1 - prob_votos)
    
    # Cálculo de AUC
    roc_knn <- roc(y_test_fold, prob_class1, levels = c("0", "1"), 
                   direction = "<", quiet = TRUE)
    auc_vals[i] <- auc(roc_knn)
  }
  
  resultados_knn <- rbind(resultados_knn, data.frame(
    k = k_val, AUC = mean(auc_vals), SD = sd(auc_vals)
  ))
  cat("K =", sprintf("%3d", k_val), "| AUC =", round(mean(auc_vals), 4), 
      "±", round(sd(auc_vals), 4), "\n")
}

# Mejor configuración
mejor_knn <- resultados_knn[which.max(resultados_knn$AUC), ]
cat("\n>>> Mejor KNN: K =", mejor_knn$k, "| AUC =", round(mejor_knn$AUC, 4), "\n")+

saveRDS(mejor_knn, "Mejor KNN")

mejor_knn

# ==============================================================================
#                              7. RDA-------------------------------------------
# ==============================================================================

# Dataframe normalizado para RDA
df_train_rda <- data.frame(X_train_scaled, Depression = y_train)
df_test_rda <- data.frame(X_test_scaled, Depression = y_test)

# Grid de hiperparámetros
lambda_grid <- c(0,0.2,0.4,0.6,0.8,1)
gamma_grid <- c(0,0.2,0.4,0.6,0.8,1)
rda_grid <- expand.grid(lambda = lambda_grid, gamma = gamma_grid)

resultados_rda <- data.frame(lambda = numeric(), gamma = numeric(), 
                              AUC = numeric(), SD = numeric())

for(g in 1:nrow(rda_grid)) {
  
  lambda_val <- rda_grid$lambda[g]
  gamma_val <- rda_grid$gamma[g]
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_fold <- df_train_rda[train_idx, ]
    test_fold <- df_train_rda[test_idx, ]
    
    tryCatch({
      # Modelo RDA
      modelo_rda <- rda(Depression ~ ., data = train_fold,
                        lambda = lambda_val, gamma = gamma_val)
      
      # Predicción
      pred_rda <- predict(modelo_rda, test_fold)
      prob_rda <- pred_rda$posterior[, "1"]
      
      # AUC
      roc_rda <- roc(test_fold$Depression, prob_rda, levels = c("0", "1"),
                     direction = "<", quiet = TRUE)
      auc_vals[i] <- auc(roc_rda)
      
    }, error = function(e) {
      auc_vals[i] <<- NA
    })
  }
  
  resultados_rda <- rbind(resultados_rda, data.frame(
    lambda = lambda_val, gamma = gamma_val,
    AUC = mean(auc_vals, na.rm = TRUE), SD = sd(auc_vals, na.rm = TRUE)
  ))
  cat("λ =", lambda_val, ", γ =", gamma_val, 
      "| AUC =", round(mean(auc_vals, na.rm = TRUE), 4), "\n")
}

# Mejor configuración
mejor_rda <- resultados_rda[which.max(resultados_rda$AUC), ]
cat("\n>>> Mejor RDA: λ =", mejor_rda$lambda, ", γ =", mejor_rda$gamma,
    "| AUC =", round(mejor_rda$AUC, 4), "\n")

saveRDS(mejor_rda, "Mejor RDA guardado")
mejor_rda


# ==============================================================================
#                         8. RED NEURONAL MULTICAPA (MLP)-----------------------
# ==============================================================================


# PREPARACIÓN DE DATOS PARA MLP


# MLP requiere que la variable respuesta sea numérica (0/1)
y_train_num <- as.numeric(as.character(y_train))
y_test_num <- as.numeric(as.character(y_test))

# Crear dataframes para nnet (usa las matrices ya normalizadas de KNN/RDA)
df_train_mlp <- data.frame(X_train_scaled, Depression = y_train_num)
df_test_mlp <- data.frame(X_test_scaled, Depression = y_test_num)

# Dimensiones de datos para MLP:

cat("Train:", nrow(df_train_mlp), "x", ncol(df_train_mlp), "\n")
cat("Test:", nrow(df_test_mlp), "x", ncol(df_test_mlp), "\n\n")


# VALIDACIÓN CRUZADA MANUAL PARA MLP


# Grid de hiperparámetros:
# - size: número de neuronas en la capa oculta
# - decay: regularización L2 (weight decay) para evitar sobreajuste

size_grid <- c(3, 5, 10, 15)
decay_grid <- c(0, 0.001, 0.01, 0.1)
mlp_grid <- expand.grid(size = size_grid, decay = decay_grid)

# Grid extendido para MLP (valores que pueden botar mejores resultados)
#size_grid <- c(2, 3, 4, 5, 7)
#decay_grid <- c(0.05, 0.1, 0.2, 0.3, 0.5)
#mlp_grid <- expand.grid(size = size_grid, decay = decay_grid)



resultados_mlp <- data.frame(size = numeric(), decay = numeric(), 
                             AUC = numeric(), SD = numeric())

# Iniciando validación cruzada para MLP...
cat("Grid: ", nrow(mlp_grid), " combinaciones de hiperparámetros\n\n")

# Usar los mismos folds que los otros modelos
for(g in 1:nrow(mlp_grid)) {
  
  size_val <- mlp_grid$size[g]
  decay_val <- mlp_grid$decay[g]
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_fold <- df_train_mlp[train_idx, ]
    test_fold <- df_train_mlp[test_idx, ]
    
    tryCatch({
      # Entrenar MLP
      # - size: neuronas en capa oculta
      # - decay: regularización L2
      # - maxit: máximo de iteraciones
      # - trace: FALSE para no imprimir progreso
      # - linout: FALSE para clasificación (usa softmax/logistic)
      
      set.seed(42)  # Reproducibilidad dentro de cada fold
      modelo_mlp <- nnet(
        Depression ~ ., 
        data = train_fold,
        size = size_val,
        decay = decay_val,
        maxit = 200,
        trace = FALSE,
        linout = FALSE  # FALSE para clasificación binaria
      )
      
      # Predicción de probabilidades
      prob_mlp <- predict(modelo_mlp, test_fold, type = "raw")[, 1]
      
      # Cálculo de AUC
      roc_mlp <- roc(test_fold$Depression, prob_mlp, levels = c(0, 1),
                     direction = "<", quiet = TRUE)
      auc_vals[i] <- auc(roc_mlp)
      
    }, error = function(e) {
      auc_vals[i] <<- NA
    })
  }
  
  resultados_mlp <- rbind(resultados_mlp, data.frame(
    size = size_val, decay = decay_val,
    AUC = mean(auc_vals, na.rm = TRUE), SD = sd(auc_vals, na.rm = TRUE)
  ))
  
  cat("size =", sprintf("%2d", size_val), ", decay =", sprintf("%.3f", decay_val),
      "| AUC =", round(mean(auc_vals, na.rm = TRUE), 4), "\n")
}

# Mejor configuración
mejor_mlp <- resultados_mlp[which.max(resultados_mlp$AUC), ]
cat("\n>>> Mejor MLP: size =", mejor_mlp$size, ", decay =", mejor_mlp$decay,
    "| AUC =", round(mejor_mlp$AUC, 4), "\n")

#-------------------------------------------------------------------------------
# Grid extendido para MLP
size_grid <- c(2, 3, 4, 5, 7)
decay_grid <- c(0.05, 0.1, 0.2, 0.3, 0.5)
mlp_grid <- expand.grid(size = size_grid, decay = decay_grid)

resultados_mlp <- data.frame(size = numeric(), decay = numeric(), 
                             AUC = numeric(), SD = numeric())

cat("Iniciando validación cruzada para MLP (grid extendido)...\n")
cat("Grid:", nrow(mlp_grid), "combinaciones\n\n")

for(g in 1:nrow(mlp_grid)) {
  
  size_val <- mlp_grid$size[g]
  decay_val <- mlp_grid$decay[g]
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_fold <- df_train_mlp[train_idx, ]
    test_fold <- df_train_mlp[test_idx, ]
    
    tryCatch({
      set.seed(42)
      modelo_mlp <- nnet(
        Depression ~ ., 
        data = train_fold,
        size = size_val,
        decay = decay_val,
        maxit = 200,
        trace = FALSE,
        linout = FALSE
      )
      
      prob_mlp <- predict(modelo_mlp, test_fold, type = "raw")[, 1]
      
      roc_mlp <- roc(test_fold$Depression, prob_mlp, levels = c(0, 1),
                     direction = "<", quiet = TRUE)
      auc_vals[i] <- auc(roc_mlp)
      
    }, error = function(e) {
      auc_vals[i] <<- NA
    })
  }
  
  resultados_mlp <- rbind(resultados_mlp, data.frame(
    size = size_val, decay = decay_val,
    AUC = mean(auc_vals, na.rm = TRUE), SD = sd(auc_vals, na.rm = TRUE)
  ))
  
  cat("size =", sprintf("%2d", size_val), ", decay =", sprintf("%.2f", decay_val),
      "| AUC =", round(mean(auc_vals, na.rm = TRUE), 4), "\n")
}

# Mejor configuración
mejor_mlp <- resultados_mlp[which.max(resultados_mlp$AUC), ]
cat("\n>>> Mejor MLP: size =", mejor_mlp$size, ", decay =", mejor_mlp$decay,
    "| AUC =", round(mejor_mlp$AUC, 4), "\n")




# ==============================================================================
#                       9. SUPPORT VECTOR MACHINE (SVM)-------------------------
# ==============================================================================

# ------------------------------------------------------------------------------
# PREPARACIÓN DE DATOS PARA SVM
# ------------------------------------------------------------------------------

# SVM requiere que la variable respuesta sea factor
# Usamos las matrices ya normalizadas (igual que KNN y RDA)
df_train_svm <- data.frame(X_train_scaled, Depression = y_train)
df_test_svm <- data.frame(X_test_scaled, Depression = y_test)

cat("Dimensiones de datos para SVM:\n")
cat("Train:", nrow(df_train_svm), "x", ncol(df_train_svm), "\n")
cat("Test:", nrow(df_test_svm), "x", ncol(df_test_svm), "\n\n")

# ------------------------------------------------------------------------------
# VALIDACIÓN CRUZADA MANUAL PARA SVM
# ------------------------------------------------------------------------------

# Grid de hiperparámetros:
# - kernel: tipo de kernel (lineal, radial, polinomial)
# - cost: parámetro de regularización C (penalización por errores)
# - gamma: parámetro del kernel radial (solo aplica para kernel="radial")

# Probaremos kernel radial (RBF) que es el más común para problemas no lineales
# y kernel lineal para comparar

# Grid para kernel RADIAL
cost_grid <- c(0.1, 1, 10)
gamma_grid <- c(0.01, 0.1, 1)
svm_grid_radial <- expand.grid(cost = cost_grid, gamma = gamma_grid)

resultados_svm <- data.frame(kernel = character(), cost = numeric(), 
                             gamma = numeric(), AUC = numeric(), SD = numeric(),
                             stringsAsFactors = FALSE)

cat("=== SVM con Kernel RADIAL (RBF) ===\n")
cat("Grid:", nrow(svm_grid_radial), "combinaciones\n\n")

# Usar los mismos folds que los otros modelos
for(g in 1:nrow(svm_grid_radial)) {
  
  cost_val <- svm_grid_radial$cost[g]
  gamma_val <- svm_grid_radial$gamma[g]
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_fold <- df_train_svm[train_idx, ]
    test_fold <- df_train_svm[test_idx, ]
    
    tryCatch({
      # Entrenar SVM con kernel radial
      modelo_svm <- svm(
        Depression ~ ., 
        data = train_fold,
        kernel = "radial",
        cost = cost_val,
        gamma = gamma_val,
        probability = TRUE  # Para obtener probabilidades
      )
      
      # Predicción con probabilidades
      pred_svm <- predict(modelo_svm, test_fold, probability = TRUE)
      prob_svm <- attr(pred_svm, "probabilities")[, "1"]
      
      # Cálculo de AUC
      roc_svm <- roc(test_fold$Depression, prob_svm, levels = c("0", "1"),
                     direction = "<", quiet = TRUE)
      auc_vals[i] <- auc(roc_svm)
      
    }, error = function(e) {
      auc_vals[i] <<- NA
    })
  }
  
  resultados_svm <- rbind(resultados_svm, data.frame(
    kernel = "radial",
    cost = cost_val, 
    gamma = gamma_val,
    AUC = mean(auc_vals, na.rm = TRUE), 
    SD = sd(auc_vals, na.rm = TRUE),
    stringsAsFactors = FALSE
  ))
  
  cat("cost =", sprintf("%5.1f", cost_val), ", gamma =", sprintf("%.2f", gamma_val),
      "| AUC =", round(mean(auc_vals, na.rm = TRUE), 4), "\n")
}

# También probamos kernel LINEAL (más simple, útil con muchas variables)
cat("\n=== SVM con Kernel LINEAL ===\n")

cost_grid_linear <- c(0.01, 0.1, 1, 10)

for(cost_val in cost_grid_linear) {
  
  auc_vals <- numeric(K)
  
  for(i in 1:K) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_fold <- df_train_svm[train_idx, ]
    test_fold <- df_train_svm[test_idx, ]
    
    tryCatch({
      # Entrenar SVM con kernel lineal
      modelo_svm <- svm(
        Depression ~ ., 
        data = train_fold,
        kernel = "linear",
        cost = cost_val,
        probability = TRUE
      )
      
      # Predicción con probabilidades
      pred_svm <- predict(modelo_svm, test_fold, probability = TRUE)
      prob_svm <- attr(pred_svm, "probabilities")[, "1"]
      
      # Cálculo de AUC
      roc_svm <- roc(test_fold$Depression, prob_svm, levels = c("0", "1"),
                     direction = "<", quiet = TRUE)
      auc_vals[i] <- auc(roc_svm)
      
    }, error = function(e) {
      auc_vals[i] <<- NA
    })
  }
  
  resultados_svm <- rbind(resultados_svm, data.frame(
    kernel = "linear",
    cost = cost_val, 
    gamma = NA,  # No aplica para kernel lineal
    AUC = mean(auc_vals, na.rm = TRUE), 
    SD = sd(auc_vals, na.rm = TRUE),
    stringsAsFactors = FALSE
  ))
  
  cat("cost =", sprintf("%5.2f", cost_val),
      "| AUC =", round(mean(auc_vals, na.rm = TRUE), 4), "\n")
}

# Mejor configuración
mejor_svm <- resultados_svm[which.max(resultados_svm$AUC), ]
cat("\n>>> Mejor SVM: kernel =", mejor_svm$kernel, 
    ", cost =", mejor_svm$cost)
if(mejor_svm$kernel == "radial") {
  cat(", gamma =", mejor_svm$gamma)
}
cat(" | AUC =", round(mejor_svm$AUC, 4), "\n")






# ==============================================================================
#                        10. ARBOL DE DECISION----------------------------------
# ==============================================================================


#Cargar librerías 
library(rpart) 
library(caret)
library(pROC)

#Se está usando los mismos folds

set.seed(1234)  # Reproducibilidad

K <- 5
n <- nrow(df_train)

# Crear folds manualmente y sin solapamiento
idx <- sample(1:n, n, replace=FALSE)
folds <- split(idx, cut(seq_along(idx), K, labels=FALSE))


# ============================================================
# Calibración de hiperparámetros 
# ============================================================
# ============================================================
#Tunear Hiperparámetro CP en base a AUC - GRID SEARCH JERÁRQUICO
# ============================================================

#ITERACION 1:

cp_grid <- seq(from = 0.001, to = 1, by = 0.09) #El grid de valores a evaluar 
resultados_arb <- data.frame(cp=cp_grid, AUC=NA, SD = NA) #Objeto para guardar resultados

for(m in seq_along(cp_grid)){ #Bucle para cada valor del grid
  cp_val <- cp_grid[m] #CP_val toma el valor del CP de la iteración actual
  auc_vals_arb <- numeric(K)   # guarda el AUC de cada iteración del fold
  
  cat("\n=========== Tunear CP =", cp_val, "===========\n")
  
  for(i in 1:K){ #Bucle para el K-Fold con K = 5
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]] #Selecciona el fold de prueba
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2) #Selecciona el resto de folds para entrenamiento
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO ARBOL DE DECISION OPTIMIZADO
    # ------------------------------
    m_arbol_cv <- rpart(
      Depression ~ ., method = "class",
      data       = train_rf2, control = rpart.control(cp = cp_val) #Se introduce el valor de cp que va a iterar
    )
    
    # Probabilidades para AUC
    prob_arb <- predict(m_arbol_cv, newdata=test_rf2, type="prob")[,2] #Se guardan las probabilidades estimadas
    
    # Cálculo AUC robusto
    roc_arb  <- roc(test_rf2$Depression, prob_arb, levels=c("0","1"), direction="<") #Se obtiene el objeto ROC
    auc_vals_arb[i] <- auc(roc_arb) #Con el objeto ROC se obtiene el AUC y se guarda
  } #Se continua con la siguiente iteración
  
  resultados_arb$AUC[m] <- mean(auc_vals_arb) #Se calcula la media del AUC de las iteraciones del K-Fold
  resultados_arb$SD[m] <- sd(auc_vals_arb) #Se calcula la desv. est. del AUC de las iteraciones del K-Fold
}
CV_TREE_x_CP_it_1 <- resultados_arb #El mejor CP está entre 0.001 y 0.091


# ITERACION 2: Grid search dentro del intervalo hallado antes

cp_grid <- seq(from = 0.001, to = 0.091, by = 0.005) #nuevos valores a evaluar
resultados_arb <- data.frame(cp=cp_grid, AUC=NA, SD = NA)
for(m in seq_along(cp_grid)){
  
  cp_val <- cp_grid[m]
  auc_vals_arb <- numeric(K)   # guarda AUC de cada fold
  
  cat("\n=========== Tunear CP =", cp_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]]
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2)
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO ARBOL DE DECISION OPTIMIZADO
    # ------------------------------
    m_arbol_cv <- rpart(
      Depression ~ ., method = "class",
      data       = train_rf2, control = rpart.control(cp = cp_val)
    )
    
    # Probabilidades para AUC
    prob_arb <- predict(m_arbol_cv, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC robusto
    roc_arb  <- roc(test_rf2$Depression, prob_arb, levels=c("0","1"), direction="<")
    auc_vals_arb[i] <- auc(roc_arb)
  }
  
  resultados_arb$AUC[m] <- mean(auc_vals_arb)
  resultados_arb$SD[m] <- sd(auc_vals_arb)
}

CV_TREE_x_CP_it_2 <- resultados_arb #El mejor intervalo se encuentra entre 0.001 y 0.006

#Iteracion 3: Grid search dentro del mejor intervalo hallado antes:

cp_grid <- seq(from = 0.001, to = 0.006, by = 0.001) #Nuevos valores a evaluar
resultados_arb <- data.frame(cp=cp_grid, AUC=NA, SD = NA)
for(m in seq_along(cp_grid)){
  
  cp_val <- cp_grid[m]
  auc_vals_arb <- numeric(K)   # guarda AUC de cada fold
  
  cat("\n=========== Tunear CP =", cp_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]]
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2)
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO ARBOL DE DECISION OPTIMIZADO
    # ------------------------------
    m_arbol_cv <- rpart(
      Depression ~ ., method = "class",
      data       = train_rf2, control = rpart.control(cp = cp_val)
    )
    
    # Probabilidades para AUC
    prob_arb <- predict(m_arbol_cv, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC robusto
    roc_arb  <- roc(test_rf2$Depression, prob_arb, levels=c("0","1"), direction="<")
    auc_vals_arb[i] <- auc(roc_arb)
  }
  
  resultados_arb$AUC[m] <- mean(auc_vals_arb)
  resultados_arb$SD[m] <- sd(auc_vals_arb)
  rm(roc_arb)
}

CV_TREE_x_CP_it_3 <- resultados_arb #Mejor AUC se da cuando CP = 0.001


# Guardar resultados: 
AUC_CV_ARBOL <- CV_TREE_x_CP_it_3[1,c(1,2)]
saveRDS(AUC_CV_ARBOL,"AUC_CV_ARBOL.RDS")
saveRDS(CV_TREE_x_CP_it_1,"CV_TREE_x_CP_it_1.RDS")
saveRDS(CV_TREE_x_CP_it_2,"CV_TREE_x_CP_it_2.RDS")
saveRDS(CV_TREE_x_CP_it_3,"CV_TREE_x_CP_it_3.RDS")


# ==============================================================================
#                        11. BAGGING--------------------------------------------
# ==============================================================================

set.seed(1234)  # Asegurar la reproducibilidad

K <- 5
n <- nrow(df_train)

# Crear folds manualmente sin solapamiento
idx <- sample(1:n, n, replace = FALSE)
folds <- split(idx, cut(seq_along(idx), K, labels = FALSE))

# ============================================================
# Calibración de hiperparámetros secuencial (nbagg y CP)
# ============================================================
# ============================================================
#Tunear Hiperparámetro nbagg en base a AUC - GRID SEARCH
# ============================================================
nbagg_vals <- c(100, 200, 300, 400, 500) #Grid de Valores a evaluar

resultados_bag <- data.frame(nbagg = nbagg_vals, AUC = NA, SD = NA) #Objeto para guardar los resultados

for(b in seq_along(nbagg_vals)){ #Bucle para cada valor del Grid
  auc_bagg <- numeric(K) #Objeto para guardar el AUC
  nb <- nbagg_vals[b] #Valor actual de nbagg en esta iteracion
  
  cat("\n==================  Evaluando nbagg =", nb, "==================\n")
  
  for(i in 1:K){ #Bucle para los folds
    
    cat("\n=========== FOLD", i, "===========\n")
    
    test_idx  <- folds[[i]] #Selección del fold de la iteracion actual para test
    train_idx <- setdiff(1:n, test_idx) #Seleccion del resto de folds para train
    
    train <- df_train[train_idx, ] 
    test  <- df_train[test_idx, ]
    
    # ------------------------------
    # Entrenar Bagging
    # ------------------------------
    set.seed(8000 + i)   # Ya que el modelo usa bootstrap se agrega una semilla que varie por iteración para que el CV sea reproducible
    modelo_bag <- bagging(
      Depression ~ ., 
      data = train,
      nbagg = nb
    )
    
    # Probabilidades para calcular AUC
    prob <- predict(modelo_bag, newdata = test, type = "prob")[,2] #Se guarda las probabilidades estimadas para el fold de test 
    
    # AUC por fold
    auc_bagg[i] <- auc(test$Depression, prob) #Se obtiene el AUC para el modelo de esta iteracion
    cat("AUC =", round(auc_bagg[i],4), "\n")
  }
  
  resultados_bag$AUC[b] <- mean(auc_bagg) #Se promedian los AUC's obtenidos por fold y se guardan para esta iteracion del grid
  resultados_bag$SD[b] <- sd(auc_bagg) #Se promedian los AUC's obtenidos por fold y se guardan para esta iteracion del grid
  cat("\n>>> AUC Promedio para nbagg =", nb, ":", round(resultados_bag$AUC[b],4), "\n")
}

cat("\n############### RESULTADOS FINALES ###############\n")
print(resultados_bag)

CV_Bagging_x_nbagg <- resultados_bag #El mejor es 500 pero por criterio de parsimonia y SD puedo elegir 100

# ============================================================
# Calibración de cp en Bagging con Grid Search Jerárquico (nbagg = 100)
# ============================================================

#Iteración 1: Encontrar el mejor intervalo
cp_grid <- seq(from = 0.001, to = 1, by = 0.09)
resultados_bag_cp <- data.frame(cp = cp_grid, AUC = NA, SD = NA)

for(c in seq_along(cp_grid)){
  
  auc_bagg <- numeric(K)
  cp_val <- cp_grid[c]
  
  cat("\n==================  Evaluando cp =", cp_val, "==================\n")
  
  for(i in 1:K){
    
    cat("\n=========== FOLD", i, "===========\n")
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train <- df_train[train_idx, ]
    test  <- df_train[test_idx, ]
    
    # ------------------------------
    # Entrenar Bagging con cp fijado en árbol base
    # ------------------------------
    set.seed(90 + i)   # Semilla para que sea reproducible
    modelo_bag <- bagging(
      Depression ~ ., 
      data = train,
      nbagg = 100,   # HALLADO EN EL TUNING ANTERIOR
      control = rpart.control(
        cp = cp_val,
        minsplit = 20,
        xval = 0
      )
    )
    
    # Probabilidades para calcular AUC
    prob <- predict(modelo_bag, newdata = test, type = "prob")[,2]
    
    # AUC por fold
    auc_bagg[i] <- auc(test$Depression, prob)
    cat("AUC =", round(auc_bagg[i],4), "\n")
  }
  
  resultados_bag_cp$AUC[c] <- mean(auc_bagg)
  resultados_bag_cp$SD[c] <- sd(auc_bagg)
  cat("\n>>> AUC Promedio para cp =", cp_val, ":", round(resultados_bag_cp$AUC[c],4), "\n")
}

cat("\n############### RESULTADOS FINALES (cp) ###############\n")
print(resultados_bag_cp) # Mejor AUC se encuentra entre 0.001 y 0.091
CV_Bagging_x_CP_iter1 <- resultados_bag_cp


# Iteración 2: Nuevo grid search ahora dentro del intervalo encontrado antes (0.001,0.091)

cp_grid <- seq(from = 0.001, to = 0.091, by = 0.01)
resultados_bag_cp <- data.frame(cp = cp_grid, AUC = NA, SD = NA)

for(c in seq_along(cp_grid)){
  
  auc_bagg <- numeric(K)
  cp_val <- cp_grid[c]
  
  cat("\n==================  Evaluando cp =", cp_val, "==================\n")
  
  for(i in 1:K){
    
    cat("\n=========== FOLD", i, "===========\n")
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train <- df_train[train_idx, ]
    test  <- df_train[test_idx, ]
    
    # ------------------------------
    # Entrenar Bagging con cp fijado en árbol base
    # ------------------------------
    set.seed(90 + i)   # semilla por fold
    modelo_bag <- bagging(
      Depression ~ ., 
      data = train,
      nbagg = 100,   # fijo
      control = rpart.control(
        cp = cp_val,
        minsplit = 20,
        xval = 0
      )
    )
    
    # Probabilidades para calcular AUC
    prob <- predict(modelo_bag, newdata = test, type = "prob")[,2]
    
    # AUC por fold
    auc_bagg[i] <- auc(test$Depression, prob)
    cat("AUC =", round(auc_bagg[i],4), "\n")
  }
  
  resultados_bag_cp$AUC[c] <- mean(auc_bagg)
  resultados_bag_cp$SD[c] <- sd(auc_bagg)
  cat("\n>>> AUC Promedio para cp =", cp_val, ":", round(resultados_bag_cp$AUC[c],4), "\n")
}

cat("\n############### RESULTADOS FINALES (cp) ###############\n")
print(resultados_bag_cp) # Mejor AUC se encuentra entre 0.001 y 0.091

CV_Bagging_x_CP_iter2 <- resultados_bag_cp # Mejor intervalo entre 0.001 y 0.011

# Iteración 3: Encontrar mejor valor de CP en el intervalo encontrado (0.001,0.011)

cp_grid <- seq(from = 0.001, to = 0.011, by = 0.001)
resultados_bag_cp <- data.frame(cp = cp_grid, AUC = NA, SD = NA)

for(c in seq_along(cp_grid)){
  
  auc_bagg <- numeric(K)
  cp_val <- cp_grid[c]
  
  cat("\n==================  Evaluando cp =", cp_val, "==================\n")
  
  for(i in 1:K){
    
    cat("\n=========== FOLD", i, "===========\n")
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train <- df_train[train_idx, ]
    test  <- df_train[test_idx, ]
    
    # ------------------------------
    # Entrenar Bagging con cp fijado en árbol base
    # ------------------------------
    set.seed(50 + i)   # semilla por fold
    modelo_bag <- bagging(
      Depression ~ ., 
      data = train,
      nbagg = 100,   # fijo
      control = rpart.control(
        cp = cp_val,
        minsplit = 20,
        xval = 0
      )
    )
    
    # Probabilidades para calcular AUC
    prob <- predict(modelo_bag, newdata = test, type = "prob")[,2]
    
    # AUC por fold
    auc_bagg[i] <- auc(test$Depression, prob)
    cat("AUC =", round(auc_bagg[i],4), "\n")
  }
  
  resultados_bag_cp$AUC[c] <- mean(auc_bagg)
  resultados_bag_cp$SD[c] <- sd(auc_bagg)
  cat("\n>>> AUC Promedio para cp =", cp_val, ":", round(resultados_bag_cp$AUC[c],4), "\n")
}

cat("\n############### RESULTADOS FINALES (cp) ###############\n")
print(resultados_bag_cp) # Mejor AUC se encuentra entre 0.001 y 0.091

CV_Bagging_x_CP_iter3 <- resultados_bag_cp # Mejor CP es 0.001 no hay otro CP dentro del SD.

AUC_CV_BAGGING <- CV_Bagging_x_CP_iter3[1,c(2)] # Ntree = 100, CP = 0.001  

#Guardar resultados
saveRDS(CV_Bagging_x_nbagg, "CV_Bagging_x_NBAGG.RDS")
saveRDS(CV_Bagging_x_CP_iter1, "CV_Bagging_x_CP_iter1.RDS")
saveRDS(CV_Bagging_x_CP_iter2, "CV_Bagging_x_CP_iter2.RDS")
saveRDS(CV_Bagging_x_CP_iter3, "CV_Bagging_x_CP_iter3.RDS")
saveRDS(AUC_CV_BAGGING, "AUC_CV_BAGGING.RDS")
# ==============================================================================
#                        12. RANDOM FOREST--------------------------------------
# ==============================================================================


library(randomForest)
library(ISLR2)
library(tree)

# ============================================================
#  CALIBRACIÓN DE HIPERPARÁMETROS Secuencial : mtry, ntree y nodesize
# ============================================================
set.seed(1234)

K <- 5
n <- nrow(df_train)

idx <- sample(1:n, n, replace=FALSE)
folds <- split(idx, cut(seq_along(idx), K, labels=FALSE))

# Hiperparámetros a evaluar
mtry_grid <- c(1,2,3,4,5,6,7,8,9,10,11,12,13)
resultados_rf2 <- data.frame(mtry=mtry_grid, AUC=NA, SD = NA)

# ============================================================
#1) CALIBRACIÓN DE MTRY. GRIDSEARCH FIJANDO NTREE = 500 Y NODESIZE = 1
# ============================================================

for(m in seq_along(mtry_grid)){ #Bucle para cada valor de mtry
  
  mtry_val <- mtry_grid[m] #El valor actual de mtry
  auc_vals_rf2 <- numeric(K) #Objeto para guardar resultados
  
  cat("\n=========== Tunear mtry =", mtry_val, "===========\n")
  
  for(i in 1:K){ #Bucle para los folds
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]] #Selecciona el fold actual para test
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2) #Usa el resto de folds para train
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # Construccion del modelo RF
    # ------------------------------
    set.seed(9000 + i)   # Ya que el modelo usa bootstrap se usa una semilla que varie por iteración para que el resultado del CV sea reproducible
    rf_model2 <- randomForest(
      Depression ~ .,
      data       = train_rf2,
      ntree      = 500, #Valor fijado
      mtry       = mtry_val,
      importance = FALSE,
      localImp   = FALSE,
      proximity  = FALSE,
      keep.forest = TRUE, 
      keep.inbag = FALSE,
      nodesize = 2 #Fijado
    )
    
    # Probabilidades para AUC
    prob_rf2 <- predict(rf_model2, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC
    roc_rf2  <- roc(test_rf2$Depression, prob_rf2, levels=c("0","1"), direction="<")
    auc_vals_rf2[i] <- auc(roc_rf2)
  }
  
  resultados_rf2$AUC[m] <- mean(auc_vals_rf2)
  resultados_rf2$SD[m] <- sd(auc_vals_rf2)
  rm(rf_model2)
}

cat("\n===== RESULTADOS RANDOM FOREST =====\n")
print(resultados_rf2)

CV_RF_x_MTRY <- resultados_rf2 #El mejor AUC es con Mtry = 2


# ============================================================
#2) CALIBRACIÓN DE NTREE. GRID SEARCH FIJANDO MTRY = 2 Y NODESIZE = 1
# ============================================================

ntree_grid <- c(100,200,300,400,500)
resultados_rf3 <- data.frame(ntree=ntree_grid, AUC=NA, SD = NA)

for(m in seq_along(ntree_grid)){
  
  ntree_val <- ntree_grid[m]
  auc_vals_rf2 <- numeric(K)
  
  cat("\n=========== Tunear ntree =", ntree_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]]
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2)
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO RANDOM FOREST OPTIMIZADO
    # ------------------------------
    set.seed(9500 + i)   # ← semilla por fold para reproducibilidad
    rf_model2 <- randomForest(
      Depression ~ .,
      data       = train_rf2,
      ntree      = ntree_val,   # hiperparámetro a tunear
      mtry       = 2,           # ya tuneado
      importance = FALSE,
      localImp   = FALSE,
      proximity  = FALSE,
      keep.forest = TRUE,
      keep.inbag = FALSE,
      nodesize = 2 #Fijado
    )
    
    # Probabilidades para AUC
    prob_rf2 <- predict(rf_model2, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC
    roc_rf2  <- roc(test_rf2$Depression, prob_rf2, levels=c("0","1"), direction="<")
    auc_vals_rf2[i] <- auc(roc_rf2)
  }
  
  resultados_rf3$AUC[m] <- mean(auc_vals_rf2)
  resultados_rf3$SD[m] <- sd(auc_vals_rf2)
  rm(rf_model2)
}

print(resultados_rf3)
CV_RF_x_NTREE <- resultados_rf3

#Mejor hiperparámetro es 400. El modelo más simple dentro de una SD es 200 asi que se toma este por parsimonia

# ============================================================
#3) CALIBRACIÓN DE NODESIZE. GRID SEARCH SECUENCIAL, FIJANDO MTRY = 2 Y NTREE = 200
# ============================================================

# Calibración 1:
node_grid <- seq(from = 1, to = floor(nrow(df_train)*4/5), by = 1200) # Para bajar costo computacional primero se vera en rangos de 1200
resultados_rf4 <- data.frame(node=node_grid, AUC=NA, SD = NA)
for(m in seq_along(node_grid)){
  
  node_val <- node_grid[m]
  auc_vals_rf2 <- numeric(K)   # guarda AUC de cada fold
  
  cat("\n=========== Tunear nodesize =", node_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]]
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2)
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO RANDOM FOREST OPTIMIZADO
    # ------------------------------
    set.seed(2500 + i)   # ← semilla por fold para reproducibilidad
    rf_model2 <- randomForest(
      Depression ~ .,
      data       = train_rf2,
      ntree      = 200,             # hiperparametro tuneado
      mtry       = 2,        # Hiperparámetro ya tuneado
      importance = F,
      localImp   = F,
      proximity  = F,
      keep.forest = T, 
      keep.inbag = F,  
      nodesize = node_val #Hiperparámetro a tunear
    )
    
    # Probabilidades para AUC
    prob_rf2 <- predict(rf_model2, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC robusto
    roc_rf2  <- roc(test_rf2$Depression, prob_rf2, levels=c("0","1"), direction="<")
    auc_vals_rf2[i] <- auc(roc_rf2)
  }
  
  resultados_rf4$AUC[m] <- mean(auc_vals_rf2)
  resultados_rf4$SD[m] <- sd(auc_vals_rf2)
  rm(rf_model2)
}
CV_RF_x_NODE1200 <- resultados_rf4 #El mejor rango está entre 1 y 1201

# Calibración 2:
node_grid <- seq(from = 1, to = 1201, by = 70)
resultados_rf4 <- data.frame(node=node_grid, AUC=NA, SD = NA)
for(m in seq_along(node_grid)){
  
  node_val <- node_grid[m]
  auc_vals_rf2 <- numeric(K)   # guarda AUC de cada fold
  
  cat("\n=========== Tunear nodesize =", node_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]]
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2)
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO RANDOM FOREST OPTIMIZADO
    # ------------------------------
    rf_model2 <- randomForest(
      Depression ~ .,
      data       = train_rf2,
      ntree      = 200,             # hiperparametro tuneado
      mtry       = 2,        # Hiperparámetro ya tuneado
      importance = F,
      localImp   = F,
      proximity  = F,
      keep.forest = T, 
      keep.inbag = F,  
      nodesize = node_val #Hiperparámetro a tunear
    )
    
    # Probabilidades para AUC
    prob_rf2 <- predict(rf_model2, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC robusto
    roc_rf2  <- roc(test_rf2$Depression, prob_rf2, levels=c("0","1"), direction="<")
    auc_vals_rf2[i] <- auc(roc_rf2)
  }
  
  resultados_rf4$AUC[m] <- mean(auc_vals_rf2)
  resultados_rf4$SD[m] <- sd(auc_vals_rf2)
  rm(rf_model2)
}
CV_RF_x_NODE70 <- resultados_rf4 #El mejor AUC esta en 71. Se evaluará todos los valores dentro de una
#desviación estandar

# Calibración 3:
node_grid <- seq(from = 1, to = 281, by = 10)
resultados_rf4 <- data.frame(node=node_grid, AUC=NA, SD = NA)
for(m in seq_along(node_grid)){
  
  node_val <- node_grid[m]
  auc_vals_rf2 <- numeric(K)   # guarda AUC de cada fold
  
  cat("\n=========== Tunear nodesize =", node_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx_rf2  <- folds[[i]]
    train_idx_rf2 <- setdiff(1:n, test_idx_rf2)
    
    train_rf2 <- df_train[train_idx_rf2, ]
    test_rf2  <- df_train[test_idx_rf2, ]
    
    # ------------------------------
    # MODELO RANDOM FOREST OPTIMIZADO
    # ------------------------------
    rf_model2 <- randomForest(
      Depression ~ .,
      data       = train_rf2,
      ntree      = 200,             # hiperparametro tuneado
      mtry       = 2,        # Hiperparámetro ya tuneado
      importance = F,
      localImp   = F,
      proximity  = F,
      keep.forest = T, 
      keep.inbag = F,  
      nodesize = node_val #Hiperparámetro a tunear
    )
    
    # Probabilidades para AUC
    prob_rf2 <- predict(rf_model2, newdata=test_rf2, type="prob")[,2]
    
    # Cálculo AUC robusto
    roc_rf2  <- roc(test_rf2$Depression, prob_rf2, levels=c("0","1"), direction="<")
    auc_vals_rf2[i] <- auc(roc_rf2)
  }
  
  resultados_rf4$AUC[m] <- mean(auc_vals_rf2)
  resultados_rf4$SD[m] <- sd(auc_vals_rf2)
  rm(rf_model2)
}
CV_RF_x_NODE_FINAL <- resultados_rf4 #El mejor numero de nodes es 31, el modelo menos complejo dentro de 
#una SD es el de 131. Así que usaremos ese
#EL MEJOR AUC:
AUC_CV_RAND.F <- CV_RF_x_NTREE[2,2] # Mtry = 2, Ntree = 200, Nodesize = 131

#GUARDAR RESULTADOS:
saveRDS(CV_RF_x_MTRY, "CV_RF_x_MTRY.RDS")
saveRDS(CV_RF_x_NTREE, "CV_RF_x_NTREE.RDS")
saveRDS(CV_RF_x_NODE_FINAL, "CV_RF_x_NODE_FINAL.RDS")
saveRDS(CV_RF_x_NODE1200, "CV_RF_x_NODE1200.RDS")
saveRDS(CV_RF_x_NODE70, "CV_RF_x_NODE70.RDS")
saveRDS(AUC_CV_RAND.F, "AUC_CV_RAND.F.RDS")




# ==============================================================================
#                        13. ADA BOOST------------------------------------------
# ==============================================================================


set.seed(1234)

K <- 5
n <- nrow(df_train)

# Crear folds manualmente
idx <- sample(1:n, n, replace=FALSE)
folds <- split(idx, cut(seq_along(idx), K, labels=FALSE))

# ============================================================
#     PASO 1: TUNEAR maxdepth con grid search, fijando mfinal = 100 y coeflearn = Freund
# ============================================================

#Hiperparámetros a evaluar en ESTA etapa
maxdepth_grid <- c(1, 2, 3, 4)
resultados_boost_depth <- data.frame(
  maxdepth = maxdepth_grid,
  AUC = NA,
  SD = NA
)

for(d in seq_along(maxdepth_grid)){ #Bucle para cada valor del grid
  
  depth_val <- maxdepth_grid[d] #Selecciona el valor actual del grid
  auc_vals_boost <- numeric(K) #Objeto para guardar resultados
  
  cat("\n=========== Tunear maxdepth =", depth_val, "===========\n")
  
  for(i in 1:K){ #Bucle para cada fold
    
    cat(" Fold", i, "\n")
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_boost <- df_train[train_idx, ]
    test_boost  <- df_train[test_idx, ]
    
    # ------------------------------
    # MODELO BOOSTING (con RPART interno)
    # ------------------------------
    set.seed(1000 + i)   # Como el modelo usa remuestreo se usa semilla dentro del bucle para que sea reproducible
    modelo_boost <- boosting(
      Depression ~ .,
      data       = train_boost,
      boos       = TRUE,
      mfinal     = 100, #Fijado
      coeflearn  = "Freund", #Fijado
      control    = rpart.control(
        maxdepth = depth_val,
        cp       = 0.01,
        minsplit = 20,
        xval     = 0
      )
    )
    
    # Probabilidades para AUC 
    pred_boost <- predict(
      modelo_boost,
      newdata = test_boost,
      type = "prob"
    )$prob[,2]
    
    roc_boost <- roc(
      test_boost$Depression,
      pred_boost,
      levels = c("0","1"),
      direction = "<"
    )
    
    auc_vals_boost[i] <- auc(roc_boost)
  }
  
  resultados_boost_depth$AUC[d] <- mean(auc_vals_boost)
  resultados_boost_depth$SD[d]  <- sd(auc_vals_boost)
  
  rm(modelo_boost)
}

cat("\n===== RESULTADOS BOOSTING (maxdepth) =====\n")
print(resultados_boost_depth)

CV_BOOST_x_MAXDEPTH <- resultados_boost_depth #El mejor valor de maxdepth es 1



# ============================================================
#     CALIBRACIÓN DE HIPERPARÁMETROS (BOOSTING - ADABAG)
#     PASO 2: TUNEAR mfinal (número de iteraciones boosting)
# ============================================================
# Se fija maxdepth óptimo encontrado antes (1) y coeflearn = Freund
# ------------------------------------------------------------
mfinal_grid <- c(50, 100, 150, 200, 300) #Grid de valores
resultados_boost_mfinal <- data.frame(
  mfinal = mfinal_grid,
  AUC = NA,
  SD = NA
)

# ============================================================
# 2) CALIBRACIÓN DE MFINAL
#    → fijando maxdepth = 1
# ============================================================

for(m in seq_along(mfinal_grid)){
  
  mfinal_val <- mfinal_grid[m]
  auc_vals_boost <- numeric(K)
  
  cat("\n=========== Tunear mfinal =", mfinal_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_boost <- df_train[train_idx, ]
    test_boost  <- df_train[test_idx, ]
    
    # ------------------------------
    # MODELO BOOSTING
    # ------------------------------
    set.seed(5000 + i)   # ← semilla por fold para reproducibilidad
    modelo_boost <- boosting(
      Depression ~ .,
      data       = train_boost,
      boos       = TRUE,
      mfinal     = mfinal_val,
      coeflearn  = "Freund",
      control    = rpart.control(
        maxdepth = 1,
        cp       = 0.01,
        minsplit = 20,
        xval     = 0
      )
    )
    
    # Probabilidades
    pred_boost <- predict(
      modelo_boost,
      newdata = test_boost,
      type = "prob"
    )$prob[,2]
    
    roc_boost <- roc(
      test_boost$Depression,
      pred_boost,
      levels = c("0","1"),
      direction = "<"
    )
    
    auc_vals_boost[i] <- auc(roc_boost)
  }
  
  resultados_boost_mfinal$AUC[m] <- mean(auc_vals_boost)
  resultados_boost_mfinal$SD[m]  <- sd(auc_vals_boost)
  
  rm(modelo_boost)
}

cat("\n===== RESULTADOS BOOSTING (mfinal) =====\n")
print(resultados_boost_mfinal)

CV_BOOST_x_MFINAL <- resultados_boost_mfinal #Mejor AUC con 200 pero por parsimonia se elige 50


# ============================================================
# 3) CALIBRACIÓN DE COEFLEARN: Grid Search
# fijando maxdepth = 1 y mfinal = 50
# ============================================================

coeflearn_grid <- c("Freund", "Breiman", "Zhu")
resultados_boost_coef <- data.frame(
  coeflearn = coeflearn_grid,
  AUC = NA,
  SD = NA
)

for(c in seq_along(coeflearn_grid)){
  
  coef_val <- coeflearn_grid[c]
  auc_vals_boost <- numeric(K)
  
  cat("\n=========== Tunear coeflearn =", coef_val, "===========\n")
  
  for(i in 1:K){
    
    cat(" Fold", i, "\n")
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(1:n, test_idx)
    
    train_boost <- df_train[train_idx, ]
    test_boost  <- df_train[test_idx, ]
    
    # ------------------------------
    # MODELO BOOSTING
    # ------------------------------
    set.seed(6000 + i)   # ← semilla por fold para reproducibilidad
    modelo_boost <- boosting(
      Depression ~ .,
      data       = train_boost,
      boos       = TRUE,
      mfinal     = 50,                   # fijo según tuning anterior
      coeflearn  = coef_val,             # parámetro que estamos tuneando
      control    = rpart.control(
        maxdepth = 1,
        cp       = 0.01,
        minsplit = 20,
        xval     = 0
      )
    )
    
    # Probabilidades
    pred_boost <- predict(
      modelo_boost,
      newdata = test_boost,
      type = "prob"
    )$prob[,2]
    
    roc_boost <- roc(
      test_boost$Depression,
      pred_boost,
      levels = c("0","1"),
      direction = "<"
    )
    
    auc_vals_boost[i] <- auc(roc_boost)
  }
  
  resultados_boost_coef$AUC[c] <- mean(auc_vals_boost)
  resultados_boost_coef$SD[c]  <- sd(auc_vals_boost)
  
  rm(modelo_boost)
}

cat("\n===== RESULTADOS BOOSTING (coeflearn) =====\n")
print(resultados_boost_coef)

CV_BOOST_x_COEFLEARN <- resultados_boost_coef #Se elige el coeficiente Breiman, por maximizar el AUC


AUC_FINAL_BOOSTING <- CV_BOOST_x_COEFLEARN[2,]
# ============================================================
#    LOS MEJORES HIPERPARAMETROS SON:
#     maxdepth = 1, mfinal = 50, coeflearn = "Breiman"
# ============================================================


# GUARDAR RESULTADOS:
saveRDS(CV_BOOST_x_COEFLEARN, "CV_BOOST_x_COEFLEARN.RDS")
saveRDS(CV_BOOST_x_MAXDEPTH, "CV_BOOST_x_MAXDEPTH.RDS")
saveRDS(CV_BOOST_x_MFINAL, "CV_BOOST_x_MFINAL.RDS")
saveRDS(AUC_FINAL_BOOSTING, "AUC_ADA_BOOSTING.RDS")



# ==============================================================================
#                        14. GRADIENT BOOST-------------------------------------
# ==============================================================================


library(gbm)
set.seed(1234)

K <- 5
n <- nrow(df_train)

idx <- sample(1:n, n, replace = FALSE)
folds <- split(idx, cut(seq_along(idx), K, labels = FALSE))

# Hiperparámetros a evaluar (primer hiperparámetro: shrinkage)
shrinkage_grid <- c(0.1, 0.05, 0.01)

resultados_gbm1 <- data.frame(
  shrinkage = shrinkage_grid,
  AUC = NA,
  SD  = NA
)

# ============================================================
# 1) CALIBRACIÓN DE SHRINKAGE CON GRID SEARCH
#    FIJANDO interaction.depth = 2, n.trees = 800
# ============================================================

for (s in seq_along(shrinkage_grid)) { #BUCLE PARA VALORES DEL GRID
  
  shrink_val <- shrinkage_grid[s]
  auc_vals_gbm1 <- numeric(K)
  
  cat("\n=========== Tunear shrinkage =", shrink_val, "===========\n")
  
  for (i in 1:K) { #BUCLE PARA LOS FOLDS
    
    cat(" Fold", i, "\n")
    
    test_idx_gbm1  <- folds[[i]]
    train_idx_gbm1 <- setdiff(1:n, test_idx_gbm1)
    
    train_gbm1 <- df_train[train_idx_gbm1, ]
    test_gbm1  <- df_train[test_idx_gbm1, ]
    
    # -------------------------------------------------
    # MODELO GBM (sin validación interna, costo reducido)
    # -------------------------------------------------
    set.seed(1975+i)
    gbm_model1 <- gbm(
      formula = as.numeric(as.character(Depression)) ~ .,
      data = train_gbm1,
      distribution = "bernoulli",
      n.trees = 800,                 
      interaction.depth = 2,         # fijo para esta etapa
      shrinkage = shrink_val,        # hiperparámetro en tuning
      bag.fraction = 1,            
      n.minobsinnode = 20,          
      train.fraction = 1,
      cv.folds = 0,
      keep.data = FALSE,
      verbose = FALSE
    )
    
    # Predicciones probabilísticas
    prob_gbm1 <- predict(
      gbm_model1,
      newdata = test_gbm1,
      n.trees = gbm_model1$n.trees,
      type = "response"
    )
    
    # Cálculo AUC
    roc_gbm1  <- roc(test_gbm1$Depression, prob_gbm1,
                     levels = c("0", "1"), direction = "<")
    auc_vals_gbm1[i] <- auc(roc_gbm1)
  }
  
  resultados_gbm1$AUC[s] <- mean(auc_vals_gbm1)
  resultados_gbm1$SD[s]  <- sd(auc_vals_gbm1)
  
  rm(gbm_model1)
}

cat("\n===== RESULTADOS GBM: SHRINKAGE =====\n")
print(resultados_gbm1)

CV_GBM_x_SHRINKAGE <- resultados_gbm1 #Mejor AUC es con Lambda = 0.05

# ============================================================
#  CALIBRACIÓN DE HIPERPARÁMETROS (GBM) — interaction.depth, CON SHRINKAJE = 0.05 Y NTREES = 800
# ============================================================

# Hiperparámetros a evaluar (segundo hiperparámetro)
interaction_grid <- c(1, 2, 3, 4)

resultados_gbm2 <- data.frame(
  interaction.depth = interaction_grid,
  AUC = NA,
  SD  = NA
)

# ============================================================
# 2) CALIBRACIÓN DE INTERACTION.DEPTH
#    FIJANDO shrinkage = 0.05, n.trees = 800
# ============================================================

for (d in seq_along(interaction_grid)) {
  
  depth_val <- interaction_grid[d]
  auc_vals_gbm2 <- numeric(K)
  
  cat("\n=========== Tunear interaction.depth =", depth_val, "===========\n")
  
  for (i in 1:K) {
    
    cat(" Fold", i, "\n")
    
    test_idx_gbm2  <- folds[[i]]
    train_idx_gbm2 <- setdiff(1:n, test_idx_gbm2)
    
    train_gbm2 <- df_train[train_idx_gbm2, ]
    test_gbm2  <- df_train[test_idx_gbm2, ]
    
    # Conversión a numérico (por seguridad)
    train_gbm2$Depression <- as.numeric(as.character(train_gbm2$Depression))
    test_gbm2$Depression  <- as.numeric(as.character(test_gbm2$Depression))
    
    # -------------------------------------------------
    # MODELO GBM (sin validación interna)
    # -------------------------------------------------
    set.seed(6+2*i)
    gbm_model2 <- gbm(
      formula = as.numeric(as.character(Depression)) ~ .,
      data = train_gbm2,
      distribution = "bernoulli",
      n.trees = 800,                # fijo
      interaction.depth = depth_val,   # hiperparámetro en tuning
      shrinkage = 0.05,             # fijo por etapa previa
      bag.fraction = 1,
      n.minobsinnode = 20,
      train.fraction = 1,
      cv.folds = 0,
      keep.data = FALSE,
      verbose = FALSE
    )
    
    # Predicciones probabilísticas
    prob_gbm2 <- predict(
      gbm_model2,
      newdata = test_gbm2,
      n.trees = gbm_model2$n.trees,
      type = "response"
    )
    
    # Cálculo AUC
    roc_gbm2  <- roc(test_gbm2$Depression, prob_gbm2,
                     levels = c(0, 1), direction = "<")
    auc_vals_gbm2[i] <- auc(roc_gbm2)
  }
  
  resultados_gbm2$AUC[d] <- mean(auc_vals_gbm2)
  resultados_gbm2$SD[d]  <- sd(auc_vals_gbm2)
  
  rm(gbm_model2)
}

cat("\n===== RESULTADOS GBM: INTERACTION.DEPTH =====\n")
print(resultados_gbm2)

CV_GBM_x_DEPTH <- resultados_gbm2 #El que maximiza el AUC es 1

# ============================================================
# 3) CALIBRACIÓN DE N.TREES
#    FIJANDO shrinkage = 0.05 y interaction.depth = 1
# ============================================================

ntrees_grid <- c(100, 200, 400, 600, 800)

resultados_gbm_ntrees <- data.frame(
  ntrees = ntrees_grid,
  AUC = NA,
  SD = NA
)

for (t in seq_along(ntrees_grid)) {
  
  ntrees_val <- ntrees_grid[t]
  auc_vals_gbm3 <- numeric(K)
  
  cat("\n=========== Tunear n.trees =", ntrees_val, "===========\n")
  
  for (i in 1:K) {
    
    cat(" Fold", i, "\n")
    
    test_idx_gbm3  <- folds[[i]]
    train_idx_gbm3 <- setdiff(1:n, test_idx_gbm3)
    
    train_gbm3 <- df_train[train_idx_gbm3, ]
    test_gbm3  <- df_train[test_idx_gbm3, ]
    
    # Conversión a numérico (seguridad)
    train_gbm3$Depression <- as.numeric(as.character(train_gbm3$Depression))
    test_gbm3$Depression  <- as.numeric(as.character(test_gbm3$Depression))
    
    # -------------------------------------------------
    # MODELO GBM (sin validación interna)
    # -------------------------------------------------
    set.seed(68+i*5)
    gbm_model3 <- gbm(
      formula = as.numeric(as.character(Depression)) ~ .,
      data = train_gbm3,
      distribution = "bernoulli",
      n.trees = ntrees_val,          # hiperparámetro en tuning
      interaction.depth = 1,         # fijo (mejor hallado)
      shrinkage = 0.05,              # fijo (mejor hallado)
      bag.fraction = 1,
      n.minobsinnode = 20,
      train.fraction = 1,
      cv.folds = 0,
      keep.data = FALSE,
      verbose = FALSE
    )
    
    # Predicciones probabilísticas
    prob_gbm3 <- predict(
      gbm_model3,
      newdata = test_gbm3,
      n.trees = gbm_model3$n.trees,
      type = "response"
    )
    
    # Cálculo AUC
    roc_gbm3  <- roc(test_gbm3$Depression, prob_gbm3,
                     levels = c(0, 1), direction = "<")
    auc_vals_gbm3[i] <- auc(roc_gbm3)
  }
  
  resultados_gbm_ntrees$AUC[t] <- mean(auc_vals_gbm3)
  resultados_gbm_ntrees$SD[t]  <- sd(auc_vals_gbm3)
  
  rm(gbm_model3)
}

cat("\n===== RESULTADOS GBM: N.TREES =====\n")
print(resultados_gbm_ntrees)

CV_GBM_NTREES <- resultados_gbm_ntrees #Mejor es 800, pero por parsimonia se elige 400 ya que es el
#menos complejo dentro de una Desviación estándar

AUC_GRAD_BOOST <- CV_GBM_NTREES[3,2] #Lambda = 0.05; N-Trees = 200; interaction.depth = 1

#Guardar resultados
saveRDS(CV_GBM_x_SHRINKAGE, "CV_GBM_x_SHRINKAGE.RDS")
saveRDS(CV_GBM_x_DEPTH, "CV_GBM_x_DEPTH.RDS")
saveRDS(CV_GBM_NTREES, "CV_GBM_NTREES.RDS")
saveRDS(AUC_GRAD_BOOST, "AUC_GRAD_BOOST.RDS")


#___________________________________________________
#GUARDAR RESULTADOS DE MODELOS DE ARBOLES y boost
#_____________________________________________________
res <- list(data.frame(AUC_ADA_BOOSTING[,2]), data.frame(AUC_CV_ARBOL[,2])
            ,data.frame(AUC_CV_BAGGING), data.frame(AUC_GRAD_BOOST), data.frame(AUC_CV_RAND.F))
saveRDS(res, "Resultados_Rubén.RDS")




# ==============================================================================
#                        15. REGRESIÓN LOGÍSTICA--------------------------------
# ==============================================================================


auc_reg_log <- numeric(K)   # guarda AUC de cada fold
df_train[,c(4,6,12)]  <- lapply(lapply(df_train[, c(4,6,12)], as.character ), as.numeric)
df_train[,c(7,8)] <- lapply(lapply(df_train[, c(7,8)], as.character), as.factor)
X <- model.matrix(Depression ~ . , data= df_train)[,-1]  # todas las dummies fijas
y <- df_train$Depression
for(i in 1:K){
  
  cat(" Fold", i, "\n")
  
  test_idx_  <- folds[[i]]
  train_idx_ <- setdiff(1:n, test_idx_)
  
  train_ <- X[train_idx_, ]
  test_  <- X[test_idx_, ]
  
  # ------------------------------
  # MODELO REGRESION LOGISTICA
  # ------------------------------
  m_logit <- glm(y[train_idx_]~train_, family = binomial(link = "logit"))
  
  # Probabilidades para AUC
  prob_log <- predict(m_logit, newdata=data.frame(test_), type = "response")
  
  # Cálculo AUC robusto
  roc_log  <- roc(y[train_idx_], prob_log, levels=c("0","1"), direction="<")
  auc_reg_log[i] <- auc(roc_log)
}
AUC_reg_log <- mean(auc_reg_log)


#Guardar resultado:
saveRDS(AUC_reg_log, "AUC_reg_log.RDS")






# ==============================================================================
#   16. EVALUACIÓN EN CONJUNTO DE TRAIN Y TEST (Regresión Logística) -----------
# ==============================================================================

#EVALUACIÓN DEL MEJOR MODELO EN ENTRENAMIENTO Y EN PRUEBA---------------------
#________________________________________________________________________________
#Construcción del mejor modelo

#Se transforman las variables ordinales de vuelta en numérica o factor
df[,c(4,6,12)]  <- lapply(lapply(df[, c(4,6,12)], as.character ), as.factor)
df[,c(7,8)] <- lapply(lapply(df[, c(7,8)], as.character), as.factor)

df2 <- model.matrix(Depression~., data = df)[,-1]
df2_train <- df2[auxtrain,]
df2_test <- df2[-auxtrain,]
train_y <- df_train$Depression
test_y <- df_test$Depression
str(df2)

#Entrenamiento del modelo en toda la data de entrenamiento:
m_final <- glm(train_y~., family = binomial(link = "logit"), data = data.frame(df2_train))

#______________________________________________________________________________________
# EVALUACIÓN DEL MEJOR MODELO EN LA DATA DE ENTRENAMIENTO: ------------------------
#____________________________________________________________________________________
pred_train <- predict(m_final, type = "response")
roc_train <- roc(train_y, pred_train, levels=c("0","1"), direction="<")
roc_train$auc #AUC 0.9242

#Importancia de variables mediante permutación:
auc_train <- roc_train$auc

perm_importance_train <- numeric(ncol(df2_test))
names(perm_importance_train) <- colnames(df2_test)

for(j in 1:ncol(df2_test)){
  
  test_x_perm <- df2_test
  test_x_perm[, j] <- sample(test_x_perm[, j])   # permutar columna j
  
  pred_perm <- predict(m_final, newdata=data.frame(test_x_perm),
                       type="response")
  
  auc_perm <- roc(test_y, pred_perm)$auc
  
  perm_importance_train[j] <- auc_train - auc_perm
}

sort(perm_importance_train, decreasing = TRUE)


#Grafico de Importancia de variables por AUC:http://127.0.0.1:31323/graphics/plot_zoom_png?width=1692&height=919

imp_df_train <- data.frame(
  Variable = names(perm_importance_train),
  DeltaAUC = as.numeric(perm_importance_train)
)

imp_plot_train <- imp_df_train %>%
  arrange(desc(DeltaAUC)) %>%
  head(20)   # top 20

ggplot(imp_plot_train, aes(x=reorder(Variable, DeltaAUC), y=DeltaAUC)) +
  geom_col() +
  coord_flip() +
  labs(
    title = " Aporte de Vars.(ΔAUC) - Train ",
    x = "Variable",
    y = "Disminución en AUC"
  ) +
  theme_minimal()

#___________________________________________________
# Evaluación del modelo en la data de prueba-----------------
#_______________________________________________
pred_final <- predict(m_final, newdata = data.frame(df2_test), type = "response")
roc_final <- roc(test_y, pred_final, levels=c("0","1"), direction="<")


#Importancia de variables mediante permutación:
auc_original <- roc_final$auc

perm_importance <- numeric(ncol(df2_test))
names(perm_importance) <- colnames(df2_test)

for(j in 1:ncol(df2_test)){
  
  test_x_perm <- df2_test
  test_x_perm[, j] <- sample(test_x_perm[, j])   # permutar columna j
  
  pred_perm <- predict(m_final, newdata=data.frame(test_x_perm),
                       type="response")
  
  auc_perm <- roc(test_y, pred_perm)$auc
  
  perm_importance[j] <- auc_original - auc_perm
}

sort(perm_importance, decreasing = TRUE)


#Grafico de Importancia de variables por AUC:
library(ggplot2)
library(dplyr)

imp_df <- data.frame(
  Variable = names(perm_importance),
  DeltaAUC = as.numeric(perm_importance)
)

imp_plot <- imp_df %>%
  arrange(desc(DeltaAUC)) %>%
  head(20)   # top 20

ggplot(imp_plot, aes(x=reorder(Variable, DeltaAUC), y=DeltaAUC)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Aporte de Vars. (ΔAUC) - Test",
    x = "Variable",
    y = "Disminución en AUC"
  ) +
  theme_minimal()

#Comparación de curvas ROC
plot.roc(roc_train, col = "blue", main = "Curvas ROC: Train vs Test") 
legend("bottomright",          # Posición (o coordenadas x, y)
       legend=c("Train", "Test"), # Etiquetas
       col=c("blue", "red"),  # Colores
       lty=1,                 # Tipo de línea (1=sólida)
       lwd=2)                 # Ancho de línea)
plot.roc(roc_final, add = T, col = 2)
abline(h = 0.95, col = 2)


# Buscando maximizar la sensibilidad (95%)
# Matriz de confusion

matriz_conf <- function(umbral){
  pred_class <- ifelse(pred_final >= umbral, "1", "0")
  pred_class <- factor(pred_class, levels=c("0","1"))
  
  library(caret)
  confusionMatrix(pred_class, test_y, positive = "1")}
matriz_conf(0.2648) #Para un umbral que obtiene sensibilidad del 95% la sensibilidad es del 64.44%, el ACC es de 82.44%







