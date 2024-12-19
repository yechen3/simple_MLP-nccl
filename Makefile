CUDA_PATH=/home/tgrogers-raid/a/common/cuda-11.7
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=train dp_train resnet

INCLUDES = -I/usr/include -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include -I/home/tgrogers-raid/a/liu2550/nccl/build/include
NVCC_FLAGS=-G --resource-usage -Xcompiler -rdynamic -Xcompiler -fopenmp -rdc=true -lnvToolsExt

IS_CUDA_11:=$(shell echo `nvcc --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0 | bc)

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L/home/tgrogers-raid/a/common/cuda-11.7/lib -lcublas -lcudnn -lgomp -lcurand -L/home/tgrogers-raid/a/liu2550/nccl/build/lib -lnccl_static
ALL_CCFLAGS += -m64 -g -std=c++11 $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES)

SRC_DIR = src
OBJ_DIR = obj

all : ${TARGET}

INCS = ${SRC_DIR}/helper.h ${SRC_DIR}/blob.h ${SRC_DIR}/blob.h ${SRC_DIR}/layer.h

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp ${INCS}
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@
${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu ${INCS}
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

${OBJ_DIR}/train.o: train.cpp ${INCS}
	@mkdir -p $(@D)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

${OBJ_DIR}/dp_train.o: dp_train.cpp ${INCS}
	@mkdir -p $(@D)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

${OBJ_DIR}/resnet.o: resnet.cpp ${INCS}
	@mkdir -p $(@D)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

OBJS = ${OBJ_DIR}/dp_train.o ${OBJ_DIR}/mnist.o ${OBJ_DIR}/loss.o ${OBJ_DIR}/layer.o ${OBJ_DIR}/network.o 

OBJS1 = ${OBJ_DIR}/train.o ${OBJ_DIR}/mnist.o ${OBJ_DIR}/loss.o ${OBJ_DIR}/layer.o ${OBJ_DIR}/network.o 

OBJS2 = ${OBJ_DIR}/resnet.o ${OBJ_DIR}/mnist.o ${OBJ_DIR}/loss.o ${OBJ_DIR}/layer.o ${OBJ_DIR}/network.o 

resnet: $(OBJS2)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

dp_train: $(OBJS)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

train: $(OBJS1)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

.PHONY: clean
clean:
	rm -f ${TARGET} ${OBJ_DIR}/*.o

