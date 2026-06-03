package com.jayendra.spring_ai.service;


import com.jayendra.spring_ai.dto.Embedding;
import com.jayendra.spring_ai.dto.Joke;
import com.jayendra.spring_ai.dto.VectorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class AIService {
    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;
    private final VectorStore vectorStore;

    public Embedding getEmbedding(String text){
        float[] em = embeddingModel.embed(text);
        return new Embedding(em.length,em);
    }

    public Joke getJoke(String topic) {

        String systemPrompt = """
                You are a sarcastic joker, you make poetic jokes in 4 lines.
                You don't make jokes about child.
                Give a joke on the topic: {topic}
                """;

        PromptTemplate promptTemplate = new PromptTemplate(systemPrompt);
        String renderedText = promptTemplate.render(Map.of("topic", topic));

        Joke response =  chatClient.prompt()
                .user(renderedText)
                .advisors(
                        new SimpleLoggerAdvisor()
                )
                .call()
                .entity(Joke.class);

        return response;
    }

    public void ingestDataToVectorStore(List<VectorData> data) {
        List<Document> documents = data.stream()
                .map(data1 -> new Document(data1.getText(),data1.getMap()))
                .toList();
        log.info("List of document ready to save");
        vectorStore.add(documents);
        log.info("List of document saved");
    }

    public List<Document> similaritySearch(String text) {
        return vectorStore.similaritySearch(SearchRequest.builder()
                .query(text)
                .topK(3)
                .similarityThreshold(0.3)
                .build());
    }
}
