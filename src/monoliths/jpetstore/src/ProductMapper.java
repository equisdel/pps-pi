package org.mybatis.jpetstore.mapper;

import java.util.List;

import org.mybatis.jpetstore.domain.Product;

/**
 * The Interface ProductMapper.
 *
 * @author Eduardo Macarron
 */
public interface ProductMapper {

  List<Product> getProductListByCategory(String categoryId);

  Product getProduct(String productId);

  List<Product> searchProductList(String keywords);

}
