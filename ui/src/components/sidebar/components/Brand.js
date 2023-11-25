import React from "react";

// Chakra imports
import {Flex, Link} from "@chakra-ui/react";

// Custom components

export function SidebarBrand() {
  return (
    <Flex align='center' direction='column'>
        <Link
            color={"currentcolor"}
            href='#'
            bg='inherit'
            borderRadius='inherit'
            marginBottom='20px'
            fontWeight='bold'
            fontSize='34px'
            _hover={{ color: "currentcolor" }}
            _active={{
                bg: 'inherit',
                transform: 'none',
                borderColor: 'transparent'
            }}
            _focus={{
                boxShadow: 'none'
            }}>
          {"NHT"}
        </Link>
    </Flex>
  );
}

export default SidebarBrand;
