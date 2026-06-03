package com.jayendra.spring_ai.dto;

import lombok.Data;

import java.util.Map;

@Data
public class VectorData {
    private String text;
    private Map<String,Object> map;
}
