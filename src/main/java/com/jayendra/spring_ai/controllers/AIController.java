package com.jayendra.spring_ai.controllers;


import com.jayendra.spring_ai.dto.Embedding;
import com.jayendra.spring_ai.dto.Joke;
import com.jayendra.spring_ai.dto.VectorData;
import com.jayendra.spring_ai.service.AIService;
import com.jayendra.spring_ai.service.RAGService;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.document.Document;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/ai")
public class AIController {

    private final AIService aiService;
    private final RAGService ragService;

    @PostMapping("/joke")
    public Joke getJoke(@RequestBody String topic){
        return aiService.getJoke(topic);
    }

    @PostMapping("/getEmbedding")
    public Embedding getEmbedding(@RequestBody String text){
        return aiService.getEmbedding(text);
    }

    @PostMapping("/ingestData")
    public void ingestDocument(@RequestBody List<VectorData> data){
        aiService.ingestDataToVectorStore(data);
    }

    @PostMapping("/similaritySearch")
    public List<Document> similaritySearch(@RequestBody String text){
        return aiService.similaritySearch(text);
    }

    @GetMapping("/ingestPdf")
    public void ingestPdf(){
        ragService.ingestPdfInVectorStore();
    }

    @PostMapping("/ask")
    public String askAI(@RequestBody String text){
        return ragService.askAI(text);
    }

    @PostMapping("/askWithAdvisor/{userId}")
    public String askAIWithAdvisor(@PathVariable String userId,@RequestBody String text){
        return ragService.askAIWithAdvisors(text,userId);
    }
}
