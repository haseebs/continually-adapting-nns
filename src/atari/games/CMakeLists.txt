add_subdirectory(supported)
target_sources(Algorithm
  PRIVATE
    Roms.cpp
    RomSettings.cpp
    RomUtils.cpp
)

#target_sources(Control
#        PRIVATE
#        Roms.cpp
#        RomSettings.cpp
#        RomUtils.cpp
#)

#target_sources(Torch
#        PRIVATE  Roms.cpp
#        RomSettings.cpp
#        RomUtils.cpp)
