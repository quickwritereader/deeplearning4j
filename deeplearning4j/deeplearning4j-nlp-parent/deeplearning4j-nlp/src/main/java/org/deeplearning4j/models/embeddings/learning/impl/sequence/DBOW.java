/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.embeddings.learning.impl.sequence;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.BatchSequences;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class DBOW<T extends SequenceElement> implements SequenceLearningAlgorithm<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;
    protected VectorsConfiguration configuration;


    protected int window;
    protected boolean useAdaGrad;
    protected double negative;

    protected SkipGram<T> skipGram = new SkipGram<>();

    private static final Logger log = LoggerFactory.getLogger(DBOW.class);

    @Override
    public ElementsLearningAlgorithm<T> getElementsLearningAlgorithm() {
        return skipGram;
    }

    public DBOW() {

    }

    @Override
    public String getCodeName() {
        return "PV-DBOW";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable,
                          @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.configuration = configuration;

        skipGram.configure(vocabCache, lookupTable, configuration);
    }

    /**
     * DBOW doesn't involve any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {

    }

    @Override
    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate,
                                BatchSequences<T> batchSequences) {

        // we just pass data to dbow, and loop over sequence there
        dbow(0, sequence, (int) nextRandom.get() % window, nextRandom, learningRate, false, null,
                batchSequences);


        return 0;
    }

    /**
     * DBOW has no reasons for early termination
     * @return
     */
    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }



    protected List<CustomOp> getOps(int i, Sequence<T> sequence, int b, AtomicLong nextRandom, double alpha, boolean isInference,
                                    INDArray inferenceVector) {


        List<T> sentence = skipGram.applySubsampling(sequence, nextRandom).getElements();
        List<CustomOp> ret = new ArrayList<>();

        if (sequence.getSequenceLabel() == null)
            return null;

        List<T> labels = new ArrayList<>();
        labels.addAll(sequence.getSequenceLabels());

        if (sentence.isEmpty() || labels.isEmpty())
            return null;

        for (T lastWord : labels) {
            for (T word : sentence) {
                if (word == null)
                    continue;

                ret.add(skipGram.iterateSampleOp(word, lastWord, nextRandom, alpha, isInference, inferenceVector));

            }
        }

        return ret;
    }


    protected void dbow(int i, Sequence<T> sequence, int b, AtomicLong nextRandom, double alpha, boolean isInference,
                        INDArray inferenceVector, BatchSequences<T> batchSequences) {

        List<T> sentence = skipGram.applySubsampling(sequence, nextRandom).getElements();


        if (sequence.getSequenceLabel() == null)
            return;

        List<T> labels = new ArrayList<>();
        labels.addAll(sequence.getSequenceLabels());

        if (sentence.isEmpty() || labels.isEmpty())
            return;

        int batchSize = configuration.getBatchSize();
        for (T lastWord : labels) {
            for (T word : sentence) {
                if (word == null)
                    continue;

                if (batchSize == 1 || batchSequences == null || isInference)
                    skipGram.iterateSample(word, lastWord, nextRandom, alpha, isInference, inferenceVector);
                else
                    batchSequences.put(word, lastWord, nextRandom.get(), alpha);
            }
        }

        if (skipGram != null && skipGram.getBatch() != null && skipGram.getBatch() != null
                && skipGram.getBatch().size() >= configuration.getBatchSize()) {
            Nd4j.getExecutioner().exec(skipGram.getBatch());
            skipGram.getBatch().clear();
        }
    }

    /**
     * This method does training on previously unseen paragraph, and returns inferred vector
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     * @return
     */
    @Override
    public INDArray inferSequence(Sequence<T> sequence, long nextRandom, double learningRate, double minLearningRate,
                                  int iterations) {
        AtomicLong nr = new AtomicLong(nextRandom);
        if (sequence.isEmpty())
            return null;




        Random random = Nd4j.getRandomFactory().getNewRandomInstance(configuration.getSeed() * sequence.hashCode(),
                lookupTable.layerSize() + 1);
        INDArray ret = Nd4j.rand(random,new long[] {1, lookupTable.layerSize()}).subi(0.5)
                .divi(lookupTable.layerSize());

        OpContext ctx = Nd4j.getExecutioner().buildContext();
        boolean setCtx = false;
        List<CustomOp> ops = new ArrayList<>();
        //build up a vector over iterations?
        for (int iter = 0; iter < iterations; iter++) {
            nr.set(Math.abs(nr.get() * 25214903917L + 11));
            List<CustomOp> ops1 = getOps(0, sequence, (int) nr.get() % window, nr, learningRate, true, ret);
            if(!setCtx) {
                ctx.setInputArrays(ops1.get(0).inputArguments());
                ctx.setOutputArrays(ops1.get(0).outputArguments());
                ctx.setBArguments(ops1.get(0).bArgs());
                setCtx = true;
            }

            ctx.setTArguments(ops1.get(0).tArgs());
            for(CustomOp customOp : ops) {
                ctx.setIArguments(customOp.iArgs());
                Nd4j.getExecutioner().exec(customOp,ctx);
            }

            learningRate = ((learningRate - minLearningRate) / (iterations - iter)) + minLearningRate;
        }






        finish();

        return ret;
    }

    @Override
    public void finish() {
        if (skipGram != null && skipGram.getBatch() != null && !skipGram.getBatch().isEmpty()) {
            Nd4j.getExecutioner().exec(skipGram.getBatch());
            skipGram.getBatch().clear();
        }
    }
}
