#pragma once
enum lightType {
    Directional,
    Spot,
    Point // no support yet
};

enum lightUpdate {
    Realtime,
    OnCreate,
};